# -*- coding: utf-8 -*-

"""
作者:王涌琦
Task C.2: BERT + 线性层
"""

# =========================
# 0. 全局参数配置
# =========================

# 地址
POS_TRAIN_PATH = r"D:\中财\第五学期\信息检索\课程项目\IR_project2\训练集\positive_trainingSet"
NEG_TRAIN_PATH = r"D:\中财\第五学期\信息检索\课程项目\IR_project2\训练集\negative_trainingSet"
TEST_PATH      = r"D:\中财\第五学期\信息检索\课程项目\IR_project2\训练集\testSet-1000.xlsx"
# 本地保存好的 BERT
BERT_LOCAL_DIR = r"D:\中财\第五学期\信息检索\课程项目\IR_project2\C\bert-base-uncased"
# 训练输出目录（模型 checkpoint、日志等）
OUTPUT_DIR     = r"D:\中财\第五学期\信息检索\课程项目\IR_project2\C\bert_finetune_outputs_4050"



# —— 训练参数 ——
RANDOM_STATE   = 666      # 随机种子，保证可复现
MAX_LENGTH     = 64       # 标题很短，64 足够
TRAIN_BATCH    = 16       # per_device_train_batch_size
EVAL_BATCH     = 32       # per_device_eval_batch_size
NUM_EPOCHS     = 1        # 20w 样本先跑 1 个 epoch 即可
LEARNING_RATE  = 2e-5     # BERT 微调常用学习率
WARMUP_RATIO   = 0.1      # 前 10% steps 用 warmup
WEIGHT_DECAY   = 0.01     # L2 正则
LOGGING_STEPS  = 200      # 每多少 step 打一次 log
FP16           = True     # 4050 支持混合精度，强烈建议开
GRAD_ACC_STEPS = 1        # 若显存不够，可改为 2

# =========================
# 1. 引入依赖
# =========================

import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)


# =========================
# 2. 读数据函数
# =========================

def load_train_data(pos_path, neg_path):
    with open(pos_path, "r", encoding="utf-8", errors="replace") as f:
        pos_lines = [line.strip() for line in f if line.strip()]

    with open(neg_path, "r", encoding="utf-8", errors="replace") as f:
        neg_lines = [line.strip() for line in f if line.strip()]

    pos_df = pd.DataFrame({"content": pos_lines, "label": 1})
    neg_df = pd.DataFrame({"content": neg_lines, "label": 0})

    train_df = pd.concat([pos_df, neg_df], ignore_index=True)

    # 去掉完全重复行
    train_df = train_df.drop_duplicates(subset=["content", "label"], keep="first")

    # 丢掉标签冲突的标题（同一个 content 既是 1 又是 0）
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


# =========================
# 3. 自定义 Dataset
# =========================

class TitlesDataset(Dataset):
    """
    给 HuggingFace Trainer 用的数据集封装：
    - texts: list[str]
    - labels: list[int] (0/1)
    """

    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text  = self.texts[idx]
        label = int(self.labels[idx])

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }
        return item


# =========================
# 4. Trainer 用的指标函数
# =========================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)

    # binary F1（以 1 为正类）
    p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1
    )

    # macro / micro F1
    _, _, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    _, _, f1_micro, _ = precision_recall_fscore_support(
        labels, preds, average="micro"
    )

    return {
        "accuracy": acc,
        "f1_binary": f1_bin,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
    }


# =========================
# 5. 主流程
# =========================

def main():
    # 1. 读取训练 & 测试集
    print("Loading training data...")
    train_df = load_train_data(POS_TRAIN_PATH, NEG_TRAIN_PATH)
    print(f"Train size (after cleaning): {len(train_df)} "
          f"(pos={sum(train_df.label==1)}, neg={sum(train_df.label==0)})")

    print("Loading test data...")
    test_df = load_test_data(TEST_PATH)
    print(f"Test size: {len(test_df)} "
          f"(pos={sum(test_df.label==1)}, neg={sum(test_df.label==0)})")

    # 2. 划分 train / dev
    train_texts, dev_texts, train_labels, dev_labels = train_test_split(
        train_df["content"].values,
        train_df["label"].values,
        test_size=0.1,
        random_state=RANDOM_STATE,
        stratify=train_df["label"].values
    )

    # 3. 加载 tokenizer & 模型（从本地 BERT 目录）
    print("Loading tokenizer and BERT model from local directory...")
    tokenizer = BertTokenizer.from_pretrained(BERT_LOCAL_DIR)

    model = BertForSequenceClassification.from_pretrained(
        BERT_LOCAL_DIR,
        num_labels=2,
        id2label={0: "NEG", 1: "POS"},
        label2id={"NEG": 0, "POS": 1}
    )

    # 4. 构建 Dataset
    train_dataset = TitlesDataset(train_texts, train_labels, tokenizer)
    dev_dataset   = TitlesDataset(dev_texts,   dev_labels,   tokenizer)
    test_dataset  = TitlesDataset(test_df["content"].values,
                                  test_df["label"].values,
                                  tokenizer)

    # 5. TrainingArguments（全部参数来自顶部配置）
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="linear",

        logging_steps=LOGGING_STEPS,
        eval_strategy="epoch",         # 每个 epoch 在 dev 上评估
        save_strategy="epoch",         # 每个 epoch 保存 checkpoint
        load_best_model_at_end=True,   # 用 dev 上 f1_macro 最好的模型
        metric_for_best_model="f1_macro",
        greater_is_better=True,

        fp16=FP16,                     # 混合精度加速
        gradient_accumulation_steps=GRAD_ACC_STEPS,

        seed=RANDOM_STATE
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # 7. 开始微调
    print("Start fine-tuning BERT...")
    trainer.train()
    print("Training finished.")

    # 8. 在 testSet-1000 上做最终评估
    print("\n========== Evaluate on testSet-1000 ==========")
    preds_output = trainer.predict(test_dataset)
    logits = preds_output.predictions
    y_test = test_df["label"].values
    y_pred = np.argmax(logits, axis=-1)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")

    # binary F1 (pos=1)
    p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1
    )
    print(f"Test F1 (binary, pos=1): {f1_bin:.4f}")

    # macro / micro / weighted F1
    for avg in ["micro", "macro", "weighted"]:
        p, r, f1_avg, _ = precision_recall_fscore_support(
            y_test, y_pred, average=avg
        )
        print(f"Test {avg.capitalize()} F1: {f1_avg:.4f}  (Precision={p:.4f}, Recall={r:.4f})")

    print("\nClassification Report (per class):")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
