# -*- coding: utf-8 -*-
"""
作者:王涌琦

任务B：使用 Word2Vec + SVM 对 CiteSeer 学术标题进行二分类（正确 / 错误）

数据说明：
1. positive_trainingSet：每行一个“正确标题”，不带后缀
2. negative_trainingSet：每行一个“错误标题”，不带后缀
3. testSet-1000.xlsx：Excel 文件，包含：
   - 'title given by manchine'：机器给出的标题
   - 'Y/N'：'Y' 表示标题正确，'N' 表示标题错误

版本号：1.0.1
升级内容:增加训练集中对冲突标题以及重复标题的处理

步骤：
1. 读取正负训练集 + 测试集
2. 对所有标题进行分词（tokenization）
3. 在所有标题上训练 Word2Vec 词向量
4. 用“平均词向量”表示每个标题
5. 使用 SVM（LinearSVC）训练分类器
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
# 0. 配置：文件路径 & 参数
# =========================

POS_TRAIN_PATH = r"D:\中财\第五学期\信息检索\课程项目\IR_project2\训练集\positive_trainingSet"
NEG_TRAIN_PATH = r"D:\中财\第五学期\信息检索\课程项目\IR_project2\训练集\negative_trainingSet"
TEST_PATH      = r"D:\中财\第五学期\信息检索\课程项目\IR_project2\训练集\testSet-1000.xlsx"

# Word2Vec 参数
W2V_VECTOR_SIZE = 50   # 词向量维度
W2V_WINDOW      = 3     # 上下文窗口大小
W2V_MIN_COUNT   = 1     # 最小词频
W2V_EPOCHS      = 30    # 训练轮数

RANDOM_STATE    = 666    # 随机种子


# wyq预设参数
TO_LOWER = False         # 是否转小写：False = 保留机器原始大小写
MIN_TOKEN_LEN = 1        # 最小 token 长度：1 表示不过滤短词（保留数字、符号）



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
    # 1. 读取正负训练集（按行 = 一个标题）
    with open(pos_path, "r", encoding="utf-8", errors="replace") as f:
        pos_lines = [line.strip() for line in f if line.strip()]

    with open(neg_path, "r", encoding="utf-8", errors="replace") as f:
        neg_lines = [line.strip() for line in f if line.strip()]

    pos_df = pd.DataFrame({"content": pos_lines, "label": 1})
    neg_df = pd.DataFrame({"content": neg_lines, "label": 0})

    train_df = pd.concat([pos_df, neg_df], ignore_index=True)

    # 2. 去掉 (content, label) 完全重复行
    train_df = train_df.drop_duplicates(subset=["content", "label"], keep="first")

    # 3. 查找“标签冲突”的标题：同一个 content 对应了 2 种 label
    label_counts = train_df.groupby("content")["label"].nunique()
    conflict_titles = label_counts[label_counts > 1].index

    # 4. 丢弃所有标签冲突的标题（既不当正也不当负）
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
    df = pd.read_excel(test_path)  # 需要 openpyxl 支持

    # 为防止列名有多余空格，这里做个统一处理
    df.columns = [c.strip() for c in df.columns]

    # 取出标题和标签
    titles = df["title given by manchine"].astype(str)
    yn_col = df["Y/N"].astype(str).str.strip()

    # 把 Y -> 1, N -> 0
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
    # 1) 是否转小写
    if TO_LOWER:
        text = text.lower()

    # 2) 使用正则把“单词”和“标点”拆开：
    #   - [A-Za-z0-9]+  匹配连续的字母数字
    #   - [^\w\s]       匹配任何非字母数字、非空白字符（比如各种标点）
    tokens = re.findall(r"[A-Za-z0-9]+|[^\w\s]", text)

    # 3) 最小长度过滤（如果你真的想过滤）
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

    all_tokenized_titles: list of list of tokens
    """
    print("Training Word2Vec model...")
    w2v_model = Word2Vec(
        sentences=all_tokenized_titles,
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        sg=1,     # sg=1 使用 skip-gram，适合小数据
        # hs=1,           
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
        # 完全 OOV，则用零向量占位
        return np.zeros(vector_size, dtype=np.float32)

    vecs = np.vstack(vecs)  # shape: (num_tokens, vector_size)
    return np.mean(vecs, axis=0)


def corpus_to_vectors(tokenized_corpus, model, vector_size):
    """
    将整个语料（多个标题）统一转换为矩阵：
    输入：tokenized_corpus: list[list[str]]
    输出：numpy array, shape = (num_titles, vector_size)
    """
    vectors = []
    for tokens in tqdm(tokenized_corpus, desc="Converting to vectors"):
        vec = title_to_vector(tokens, model, vector_size)
        vectors.append(vec)
    return np.vstack(vectors)


# =========================
# 5. 训练 SVM & 评估
# =========================

def train_and_evaluate_svm(train_X, train_y, test_X, test_y):
    """
    使用 LinearSVC 训练 SVM 分类器，并在测试集上评估。
    输出：Accuracy、Precision、Recall、F1、Macro/Micro F1 等。
    """
    from sklearn.svm import LinearSVC

    print("Training SVM classifier (LinearSVC)...")
    clf = LinearSVC(random_state=RANDOM_STATE)
    clf.fit(train_X, train_y)

    print("SVM training finished. Evaluating on test set...")

    y_pred = clf.predict(test_X)


    print("================ taskB result ================")
    # 1) Accuracy
    acc = accuracy_score(test_y, y_pred)
    print(f"\nAccuracy: {acc:.4f}")

    # 2) 以正类=1 为主，输出二分类的 precision/recall/f1
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_y, y_pred, average="binary", pos_label=1
    )
    print(f"Precision (pos=1): {precision:.4f}")
    print(f"Recall    (pos=1): {recall:.4f}")
    print(f"F1-score  (pos=1): {f1:.4f}")

    # 3) Macro / Micro / Weighted F1
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
    # 1. 读取训练集
    print("Loading training data...")
    train_df = load_train_data(POS_TRAIN_PATH, NEG_TRAIN_PATH)
    print(f"Train size: {len(train_df)} "
          f"(pos={sum(train_df.label==1)}, neg={sum(train_df.label==0)})")

    # 2. 读取测试集
    print("Loading test data...")
    test_df = load_test_data(TEST_PATH)
    print(f"Test size: {len(test_df)} "
          f"(pos={sum(test_df.label==1)}, neg={sum(test_df.label==0)})")

    # 3. 合并所有标题，统一分词用于训练 Word2Vec
    all_texts = pd.concat(
        [train_df["content"], test_df["content"]],
        ignore_index=True
    )
    all_tokenized = tokenize_corpus(all_texts)

    # 4. 训练 Word2Vec
    w2v_model = train_word2vec(all_tokenized)

    # 5. 拆回训练 & 测试的 token 序列
    train_tokenized = all_tokenized[: len(train_df)]
    test_tokenized  = all_tokenized[len(train_df):]

    # 6. 将所有标题转换为向量
    train_vectors = corpus_to_vectors(train_tokenized, w2v_model, W2V_VECTOR_SIZE)
    test_vectors  = corpus_to_vectors(test_tokenized,  w2v_model, W2V_VECTOR_SIZE)

    train_labels = train_df["label"].values
    test_labels  = test_df["label"].values


    # 7. 训练 SVM 并评估
    clf = train_and_evaluate_svm(train_vectors, train_labels,
                                 test_vectors, test_labels)

    # 8. （可选）保存模型，方便后续加载
    # from joblib import dump
    # w2v_model.save("w2v_model_titles.bin")
    # dump(clf, "svm_title_classifier.joblib")
    # print("Models saved to disk.")


if __name__ == "__main__":
    main()
