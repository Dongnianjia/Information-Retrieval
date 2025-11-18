import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings("ignore")

# ===================== #
# Step 0. 自动检测 GPU
# ===================== #
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] 当前设备：{device}")
if device == "cpu":
    print("[WARN] 未检测到 GPU，训练速度可能较慢。")

# ===================== #
# Step 1. 数据加载
# ===================== #
def read_all_lines(path):
    """逐行读取文本文件，保留所有符号"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]
    return pd.DataFrame({"title": lines})

pos_path = "/private/home/wuhao/xinxijiansuo/positive_trainingSet.txt"
neg_path = "/private/home/wuhao/xinxijiansuo/negative_trainingSet.txt"
test_path = "/private/home/wuhao/xinxijiansuo/testSet-1000.xlsx"

pos = read_all_lines(pos_path)
neg = read_all_lines(neg_path)
pos["label"], neg["label"] = 1, 0
train_df = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=42)

test_df = pd.read_excel(test_path)
test_df.rename(columns={"title given by manchine": "title"}, inplace=True)
test_df["label"] = test_df["Y/N"].map({"Y": 1, "N": 0})

# 修复缺失标题
mask = test_df["title"].isna() & (test_df["label"] == 1)
test_df.loc[mask, "title"] = test_df.loc[mask, "original title"]
test_df["title"] = test_df["title"].fillna("").astype(str)

print(f"[INFO] 成功加载数据：训练集 {len(train_df)} 条，测试集 {len(test_df)} 条。")

# ===================== #
# Step 2. 下载并加载预训练模型
# ===================== #
print("[INFO] 检查预训练模型 'bert-base-uncased' 是否可用...")
from transformers import logging
logging.set_verbosity_error()

try:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    print("[INFO] 模型已成功加载 ✅")
except Exception as e:
    print("[ERROR] 模型加载失败，请检查网络或路径。错误详情：", e)
    exit()

model.to(device)

# ===================== #
# Step 3. 构建 Dataset
# ===================== #
class TitleDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=64):
        self.texts = df["title"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = torch.tensor(label, dtype=torch.long)
        return encoding

train_ds = TitleDataset(train_df, tokenizer)
test_ds = TitleDataset(test_df, tokenizer)

# ===================== #
# Step 4. 训练配置
# ===================== #
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

args = TrainingArguments(
    output_dir="./bert_finetune_output",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=50,
    fp16=True,  # 混合精度训练
    load_best_model_at_end=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

# ===================== #
# Step 5. 训练与评估
# ===================== #
print("[INFO] 开始训练 BERT 模型 ...")
trainer.train()
metrics = trainer.evaluate()

print("\n=== Evaluation Metrics ===")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

trainer.save_model("./bert_finetune_best")
print("\n[INFO] 最优模型已保存到 ./bert_finetune_best ✅")
