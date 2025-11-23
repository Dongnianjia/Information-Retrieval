# -*- coding: utf-8 -*-
'''
作者:王涌琦
任务C： BERT + SVM 
'''
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from sklearn.svm import LinearSVC

import torch
from transformers import BertTokenizer, BertModel

# =========================
# 0. 配置：路径 & 参数
# =========================

POS_TRAIN_PATH = r"D:\中财\第五学期\信息检索\课程项目\IR_project2\训练集\positive_trainingSet"
NEG_TRAIN_PATH = r"D:\中财\第五学期\信息检索\课程项目\IR_project2\训练集\negative_trainingSet"
TEST_PATH      = r"D:\中财\第五学期\信息检索\课程项目\IR_project2\训练集\testSet-1000.xlsx"

# 刚才保存 BERT 的本地目录
BERT_LOCAL_DIR = r"D:\中财\第五学期\信息检索\课程项目\IR_project2\C\bert-base-uncased"

RANDOM_STATE = 666
MAX_LENGTH   = 64    
BATCH_SIZE   = 32


def load_train_data(pos_path, neg_path):
    with open(pos_path, "r", encoding="utf-8", errors="replace") as f:
        pos_lines = [line.strip() for line in f if line.strip()]

    with open(neg_path, "r", encoding="utf-8", errors="replace") as f:
        neg_lines = [line.strip() for line in f if line.strip()]

    pos_df = pd.DataFrame({"content": pos_lines, "label": 1})
    neg_df = pd.DataFrame({"content": neg_lines, "label": 0})

    train_df = pd.concat([pos_df, neg_df], ignore_index=True)

    train_df = train_df.drop_duplicates(subset=["content", "label"], keep="first")

    label_counts = train_df.groupby("content")["label"].nunique()
    conflict_titles = label_counts[label_counts > 1].index
    if len(conflict_titles) > 0:
        train_df = train_df[~train_df["content"].isin(conflict_titles)]

    return train_df


def load_test_data(test_path):
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

class BertSentenceEncoder(object):
    """
    用本地的 bert-base-uncased 把标题编码成句向量（默认取 [CLS] 向量）
    """

    def __init__(self,
                 model_dir=BERT_LOCAL_DIR,
                 max_length=MAX_LENGTH,
                 device=None):
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertModel.from_pretrained(model_dir)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length

    @torch.no_grad()
    def encode(self, texts, batch_size=BATCH_SIZE):
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding with BERT"):
            batch_texts = texts[i:i + batch_size]

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask)

            # 取 [CLS] 向量
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(cls_embeddings.cpu().numpy())

        all_embeddings = np.vstack(all_embeddings)
        return all_embeddings


def train_and_evaluate_svm(train_X, train_y, test_X, test_y):
    print("Training SVM classifier (LinearSVC)...")
    clf = LinearSVC(random_state=RANDOM_STATE)
    clf.fit(train_X, train_y)

    print("SVM training finished. Evaluating on test set...")
    y_pred = clf.predict(test_X)

    print("================ taskC (BERT + SVM) result ================")
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


def main():
    print("Loading training data...")
    train_df = load_train_data(POS_TRAIN_PATH, NEG_TRAIN_PATH)
    print(f"Train size: {len(train_df)} "
          f"(pos={sum(train_df.label==1)}, neg={sum(train_df.label==0)})")

    print("Loading test data...")
    test_df = load_test_data(TEST_PATH)
    print(f"Test size: {len(test_df)} "
          f"(pos={sum(test_df.label==1)}, neg={sum(test_df.label==0)})")

    # 初始化本地 BERT 编码器
    encoder = BertSentenceEncoder()

    # 编码成句向量
    X_train = encoder.encode(train_df["content"].tolist())
    y_train = train_df["label"].values

    X_test = encoder.encode(test_df["content"].tolist())
    y_test = test_df["label"].values

    print("Train embeddings shape:", X_train.shape)
    print("Test  embeddings shape:", X_test.shape)

    # 训练 + 评估
    clf = train_and_evaluate_svm(X_train, y_train, X_test, y_test)

    # 可选：保存 SVM 模型
    # from joblib import dump
    # dump(clf, r"D:\中财\第五学期\信息检索\课程项目\IR_project2\C\bert_svm_title_classifier.joblib")
    # print("SVM model saved.")


if __name__ == "__main__":
    main()
