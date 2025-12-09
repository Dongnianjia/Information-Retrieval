# -*- coding: utf-8 -*-
"""
任务B：使用 Word2Vec + SVM 对 CiteSeer 学术标题进行二分类（正确 / 错误）

数据说明：
1. positive_trainingSet：每行一个“正确标题”，不带后缀
2. negative_trainingSet：每行一个“错误标题”，不带后缀
3. testSet-1000.xlsx：Excel 文件，包含：
   - 'title given by manchine'：机器给出的标题
   - 'Y/N'：'Y' 表示标题正确，'N' 表示标题错误

版本号：2.0.0
升级内容: 增加SVM训练方法,使用 SVC (C-SVC) 而不是 LinearSVC

步骤：
1. 读取正负训练集 + 测试集
2. 对所有标题进行分词（tokenization）
3. 在所有标题上训练 Word2Vec 词向量
4. 用“平均词向量”表示每个标题
5. 使用 SVM（SVC / C-SVC，可选核函数）训练分类器
6. 在测试集上评估性能，输出 Accuracy、Precision、Recall、F1、Macro/Micro F1 等指标
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from gensim.models import Word2Vec
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# =========================
# 0. 配置：文件路径 & 超参数
# =========================

# 如果你的脚本和数据在同一目录，下面这样即可；
# 否则改成绝对路径，如 "D:/xxx/positive_trainingSet"
POS_TRAIN_PATH = r"positive_trainingSet.txt"
NEG_TRAIN_PATH = r"negative_trainingSet.txt"
TEST_PATH      = r"testSet-1000.xlsx"

# Word2Vec 超参数
W2V_VECTOR_SIZE = 190   # 词向量维度
W2V_WINDOW      = 3     # 上下文窗口大小
W2V_MIN_COUNT   = 1     # 最小词频（标题短，用1比较安全）
W2V_EPOCHS      = 20   # 训练轮数

RANDOM_STATE    = 666   # 随机种子（为了结果可复现）

# wyq预设参数
TO_LOWER = False        # 是否转小写：False = 保留机器原始大小写
MIN_TOKEN_LEN = 1       # 最小 token 长度：1 表示不过滤短词（保留数字、符号）

# =========================
# 0.1 SVC (C-SVC) 关键参数（全部集中在这里）
# =========================
# 你老师要求“使用 SVM”，且你已验证非线性更好，
# 这里默认用 RBF（这是 sklearn 的 SVC 默认核）
SVC_KERNEL = "rbf"          # "rbf" / "linear" / "poly" / "sigmoid" / "precomputed"

# 软间隔强度
SVC_C = 1

# RBF / poly / sigmoid 相关
# 建议优先用 "scale"（更稳）
SVC_GAMMA = "scale"         # "scale" / "auto" / float

# 仅 poly 有意义
SVC_DEGREE = 3

# poly / sigmoid 有意义
SVC_COEF0 = 0.0

# 训练策略 & 工程参数
SVC_SHRINKING = True
SVC_PROBABILITY = False     # True 会明显变慢（不做概率时建议 False）
SVC_TOL = 1e-2
SVC_CACHE_SIZE = 8000       # MB，内存允许的话可以再加
SVC_CLASS_WEIGHT = None     # "balanced" 或 dict，如 {0:1, 1:2}
SVC_MAX_ITER = 100000           # -1 表示不限制
SVC_DECISION_FUNCTION_SHAPE = "ovr"   # 二分类其实无所谓
SVC_VERBOSE = False

# 是否对输入向量做标准化
# 对 RBF 等非线性核非常推荐
USE_STANDARD_SCALER = True


# =========================
# 1. 数据读取函数
# =========================

def load_train_data(pos_path, neg_path):
    """
    读取正负训练集，返回一个 DataFrame：
        - content: 标题文本
        - label:   1 表示正确标题（正样本）
                   0 表示错误标题（负样本）

    清洗步骤：
    1）逐行读取，去掉空行
    2）合并正负样本
    3）去掉 (content, label) 完全重复行
    4）丢弃“标签冲突”的标题：
       同一个 content 同时出现在 label=1 和 label=0 中的，全丢弃，避免噪音
    """
    with open(pos_path, "r", encoding="utf-8", errors="replace") as f:
        pos_lines = [line.strip() for line in f if line.strip()]

    with open(neg_path, "r", encoding="utf-8", errors="replace") as f:
        neg_lines = [line.strip() for line in f if line.strip()]

    pos_df = pd.DataFrame({"content": pos_lines, "label": 1})
    neg_df = pd.DataFrame({"content": neg_lines, "label": 0})

    train_df = pd.concat([pos_df, neg_df], ignore_index=True)

    # 去掉 (content, label) 完全重复行
    train_df = train_df.drop_duplicates(subset=["content", "label"], keep="first")

    # 查找“标签冲突”的标题
    label_counts = train_df.groupby("content")["label"].nunique()
    conflict_titles = label_counts[label_counts > 1].index

    # 丢弃所有标签冲突的标题
    if len(conflict_titles) > 0:
        train_df = train_df[~train_df["content"].isin(conflict_titles)]

    return train_df


def load_test_data(test_path):
    """
    读取 testSet-1000.xlsx，使用两列：
        - 'title given by manchine'：机器标题
        - 'Y/N'：'Y' -> 1（正确），'N' -> 0（错误）

    返回 DataFrame：
        - content: 标题文本
        - label:   0/1
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
# 2. 文本分词（针对英文标题）
# =========================

import re

def tokenize_text(text):
    """
    改进版 tokenizer：
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
    """
    对一个 pandas Series（多条标题）进行分词。
    返回：list[list[str]]，每条标题 = 一个 token list。
    """
    tokenized = []
    for text in tqdm(text_series, desc="Tokenizing titles"):
        tokenized.append(tokenize_text(text))
    return tokenized


# =========================
# 3. 训练 Word2Vec 模型
# =========================

def train_word2vec(all_tokenized_titles):
    """
    在所有标题（训练 + 测试）上训练 Word2Vec，
    这样可以减少测试集里 OOV（未登录词）。
    """
    print("Training Word2Vec model...")
    w2v_model = Word2Vec(
        sentences=all_tokenized_titles,
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        sg=1,
        workers=4,
        epochs=W2V_EPOCHS,
        seed=RANDOM_STATE
    )
    print("Word2Vec training finished.")
    return w2v_model


# =========================
# 4. 标题 -> 向量（平均词向量）
# =========================

def title_to_vector(tokens, model, vector_size):
    """
    将一条标题（tokens 列表）转换为一个固定维度的向量。
    做法：取每个 token 的词向量，然后求平均（average pooling）。
    如果所有 token 都不在模型词表中，则返回全零向量。
    """
    vecs = []
    for w in tokens:
        if w in model.wv:
            vecs.append(model.wv[w])

    if not vecs:
        return np.zeros(vector_size, dtype=np.float32)

    vecs = np.vstack(vecs)
    return np.mean(vecs, axis=0)


def corpus_to_vectors(tokenized_corpus, model, vector_size):
    """
    将整个语料（多个标题）统一转换为矩阵：
    输出：numpy array, shape = (num_titles, vector_size)
    """
    vectors = []
    for tokens in tqdm(tokenized_corpus, desc="Converting to vectors"):
        vec = title_to_vector(tokens, model, vector_size)
        vectors.append(vec)
    return np.vstack(vectors)


# =========================
# 5. 训练 SVM (SVC/C-SVC) & 评估
# =========================

def build_svc_model():
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    svc = SVC(
        kernel=SVC_KERNEL,
        C=SVC_C,
        gamma=SVC_GAMMA,
        degree=SVC_DEGREE,
        coef0=SVC_COEF0,
        shrinking=SVC_SHRINKING,
        probability=SVC_PROBABILITY,
        tol=SVC_TOL,
        cache_size=SVC_CACHE_SIZE,
        class_weight=SVC_CLASS_WEIGHT,
        max_iter=SVC_MAX_ITER,
        decision_function_shape=SVC_DECISION_FUNCTION_SHAPE,
        verbose=SVC_VERBOSE,
        random_state=RANDOM_STATE
    )

    if USE_STANDARD_SCALER:
        return make_pipeline(StandardScaler(), svc)
    return svc


def train_and_evaluate_svm(train_X, train_y, test_X, test_y):
    """
    使用 SVC (C-SVC) 训练分类器，并在测试集上评估。
    输出：Accuracy、Precision、Recall、F1、Macro/Micro F1 等。
    """
    print("Training SVM classifier (SVC / C-SVC)...")
    clf = build_svc_model()
    clf.fit(train_X, train_y)

    print("SVM training finished. Evaluating on test set...")

    y_pred = clf.predict(test_X)

    print("================ taskB result ================")
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

    return clf


# =========================
# 6. 主流程
# =========================

def main():
    print("Loading training data...")
    train_df = load_train_data(POS_TRAIN_PATH, NEG_TRAIN_PATH)
    print(f"Train size: {len(train_df)} "
          f"(pos={sum(train_df.label==1)}, neg={sum(train_df.label==0)})")

    print("Loading test data...")
    test_df = load_test_data(TEST_PATH)
    print(f"Test size: {len(test_df)} "
          f"(pos={sum(test_df.label==1)}, neg={sum(test_df.label==0)})")

    all_texts = pd.concat(
        [train_df["content"], test_df["content"]],
        ignore_index=True
    )
    all_tokenized = tokenize_corpus(all_texts)

    w2v_model = train_word2vec(all_tokenized)

    train_tokenized = all_tokenized[: len(train_df)]
    test_tokenized  = all_tokenized[len(train_df):]

    train_vectors = corpus_to_vectors(train_tokenized, w2v_model, W2V_VECTOR_SIZE)
    test_vectors  = corpus_to_vectors(test_tokenized,  w2v_model, W2V_VECTOR_SIZE)

    train_labels = train_df["label"].values
    test_labels  = test_df["label"].values

    clf = train_and_evaluate_svm(
        train_vectors, train_labels,
        test_vectors, test_labels
    )


    # =========================
    # 7. 保存模型（新增，不改动你原有代码）
    # =========================
    import os
    from joblib import dump

    save_dir = "models_taskBv2.0.0"
    os.makedirs(save_dir, exist_ok=True)

    # 1) 保存 Word2Vec
    w2v_path = os.path.join(save_dir, f"w2v_titles_dim{W2V_VECTOR_SIZE}.bin")
    w2v_model.save(w2v_path)

    # 2) 保存 SVC（注意：如果你开启了 StandardScaler，这里保存的是 Pipeline）
    svm_path = os.path.join(save_dir, f"svc_titles_{SVC_KERNEL}_dim{W2V_VECTOR_SIZE}.joblib")
    dump(clf, svm_path)

    print(f"\n[Saved] Word2Vec -> {w2v_path}")
    print(f"[Saved] SVC/Pipeline -> {svm_path}")


if __name__ == "__main__":
    main()
