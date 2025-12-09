# -*- coding: utf-8 -*-
"""
任务B - 推理/评估脚本：
加载本地保存的 Word2Vec + SVC(Pipeline) 模型，
在 testSet-1000.xlsx 上直接预测并输出指标。

请确保你之前保存模型的代码已生成：
./models_taskB/
    - w2v_titles_dim190.bin
    - svc_titles_rbf_dim190.joblib
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec
from joblib import load

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# =========================
# 0. 路径 & 选择要加载的模型版本
# =========================

TEST_PATH = r"testSet-1000.xlsx"

# 你保存模型的目录
MODEL_DIR = "models_taskBv2.0.0"

# 你要加载的模型“标识”
# 必须与你保存时的命名一致
LOAD_W2V_DIM = 190
LOAD_SVC_KERNEL = "rbf"

W2V_PATH = os.path.join(MODEL_DIR, f"w2v_titles_dim{LOAD_W2V_DIM}.bin")
SVC_PATH = os.path.join(MODEL_DIR, f"svc_titles_{LOAD_SVC_KERNEL}_dim{LOAD_W2V_DIM}.joblib")

# 分词设置（要和训练时一致）
TO_LOWER = False
MIN_TOKEN_LEN = 1


# =========================
# 1. 读取测试集
# =========================

def load_test_data(test_path):
    """
    读取 testSet-1000.xlsx，使用两列：
        - 'title given by manchine'：机器标题
        - 'Y/N'：'Y' -> 1（正确），'N' -> 0（错误）
    """
    df = pd.read_excel(test_path)
    df.columns = [c.strip() for c in df.columns]

    titles = df["title given by manchine"].astype(str)
    yn_col = df["Y/N"].astype(str).str.strip()

    labels = yn_col.map(lambda x: 1 if x.upper() == "Y" else 0)

    test_df = pd.DataFrame({
        "content": titles,
        "label": labels
    })
    return test_df


# =========================
# 2. 分词（保留标点）
# =========================

import re

def tokenize_text(text):
    """
    与训练脚本一致：
    - 可选是否转小写（TO_LOWER）
    - 可选最小长度过滤（MIN_TOKEN_LEN）
    - 保留标点、数字作为单独 token
    """
    if TO_LOWER:
        text = text.lower()

    tokens = re.findall(r"[A-Za-z0-9]+|[^\w\s]", text)

    if MIN_TOKEN_LEN > 1:
        tokens = [t for t in tokens if len(t) >= MIN_TOKEN_LEN]

    return tokens


def tokenize_corpus(text_series):
    tokenized = []
    for text in tqdm(text_series, desc="Tokenizing titles"):
        tokenized.append(tokenize_text(text))
    return tokenized


# =========================
# 3. 标题 -> 向量（平均词向量）
# =========================

def title_to_vector(tokens, model, vector_size):
    vecs = []
    for w in tokens:
        if w in model.wv:
            vecs.append(model.wv[w])

    if not vecs:
        return np.zeros(vector_size, dtype=np.float32)

    vecs = np.vstack(vecs)
    return np.mean(vecs, axis=0)


def corpus_to_vectors(tokenized_corpus, model, vector_size):
    vectors = []
    for tokens in tqdm(tokenized_corpus, desc="Converting to vectors"):
        vec = title_to_vector(tokens, model, vector_size)
        vectors.append(vec)
    return np.vstack(vectors)


# =========================
# 4. 评估函数
# =========================

def evaluate(test_y, y_pred):
    print("================ taskB inference result ================")

    acc = accuracy_score(test_y, y_pred)
    print(f"\nAccuracy: {acc:.4f}")

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_y, y_pred, average="binary", pos_label=1
    )
    print(f"Precision (pos=1): {precision:.4f}")
    print(f"Recall    (pos=1): {recall:.4f}")
    print(f"F1-score  (pos=1): {f1:.4f}")

    for avg in ["micro", "macro", "weighted"]:
        p, r, f1_avg, _ = precision_recall_fscore_support(
            test_y, y_pred, average=avg
        )
        print(f"{avg.capitalize()} F1: {f1_avg:.4f}  (Precision={p:.4f}, Recall={r:.4f})")

    print("\nClassification Report (per class):")
    print(classification_report(test_y, y_pred, digits=4))

    print("Confusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(test_y, y_pred))


# =========================
# 5. 主流程
# =========================

def main():
    # 0) 检查模型文件
    if not os.path.exists(W2V_PATH):
        raise FileNotFoundError(f"Word2Vec model not found: {W2V_PATH}")
    if not os.path.exists(SVC_PATH):
        raise FileNotFoundError(f"SVC/Pipeline model not found: {SVC_PATH}")

    # 1) 加载模型
    print(f"Loading Word2Vec from: {W2V_PATH}")
    w2v_model = Word2Vec.load(W2V_PATH)

    print(f"Loading SVC/Pipeline from: {SVC_PATH}")
    clf = load(SVC_PATH)

    # 2) 读取测试集
    print("Loading test data...")
    test_df = load_test_data(TEST_PATH)

    # 3) 分词
    test_tokenized = tokenize_corpus(test_df["content"])

    # 4) 向量化
    vector_size = w2v_model.wv.vector_size
    test_vectors = corpus_to_vectors(test_tokenized, w2v_model, vector_size)

    test_labels = test_df["label"].values

    # 5) 预测 + 指标
    print("Predicting on test set...")
    y_pred = clf.predict(test_vectors)

    evaluate(test_labels, y_pred)


if __name__ == "__main__":
    main()
