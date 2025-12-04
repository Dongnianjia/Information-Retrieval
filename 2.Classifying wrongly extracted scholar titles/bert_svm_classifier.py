import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ====================================================
# Step 0. 自动检测 GPU
# ====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] 当前设备：{device}")

# ====================================================
# Step 1. 数据加载
# ====================================================
def read_all_lines(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]
    return pd.DataFrame({"title": lines})

pos_path = r"D:\homework\大三上\信息检索\作业2\positive_trainingSet.txt"
neg_path = r"D:\homework\大三上\信息检索\作业2\negative_trainingSet.txt"
test_path = r"D:\homework\大三上\信息检索\作业2\testSet-1000.xlsx"

pos = read_all_lines(pos_path)
neg = read_all_lines(neg_path)
pos["label"], neg["label"] = 1, 0
train_df = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=42)

test_df = pd.read_excel(test_path)
test_df.rename(columns={"title given by manchine": "title"}, inplace=True)
test_df["label"] = test_df["Y/N"].map({"Y": 1, "N": 0})

mask = test_df["title"].isna() & (test_df["label"] == 1)
test_df.loc[mask, "title"] = test_df.loc[mask, "original title"]
test_df["title"] = test_df["title"].fillna("").astype(str)

print(f"[INFO] 数据：训练 {len(train_df)}，测试 {len(test_df)}")

# ====================================================
# Step 2. 加载模型 + Multi-Sample Dropout 分类头
# ====================================================
MODEL_NAME = "prajjwal1/bert-medium"
print(f"[INFO] 加载模型: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
config = AutoConfig.from_pretrained(MODEL_NAME)
config.num_labels = 2

# ---- 自定义模型：加入 Multi-Sample Dropout ----
class MSD_BERT(nn.Module):
    def __init__(self, model_name, num_labels=2, dropout_rate=0.5, drops=5):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.drops = drops

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooler = out.last_hidden_state[:, 0]  # CLS 向量

        logits = 0
        for _ in range(self.drops):
            logits += self.classifier(self.dropout(pooler))
        logits = logits / self.drops

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)  # label smoothing
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}

model = MSD_BERT(MODEL_NAME).to(device)

# ====================================================
# Step 3. Dataset
# ====================================================
class TitleDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=16):
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

# ====================================================
# Step 4. Metrics
# ====================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# ====================================================
# Step 5. Training Args
# ====================================================
args = TrainingArguments(
    output_dir="./tinybert_output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="none",
    save_safetensors=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

# ====================================================
# Step 6. Train & Evaluate
# ====================================================
print("[INFO] 开始训练 ...")
trainer.train()

metrics = trainer.evaluate()
print("\n=== Evaluation Metrics ===")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

trainer.save_model("./tinybert_best")
print("[INFO] 最优模型已保存到 ./tinybert_best")

# ====================================================
# Step 7. t-SNE 可视化
# ====================================================
model.eval()

backbone = model.bert  # 多样本 dropout 结构中，backbone 是 model.bert

@torch.no_grad()
def get_sentence_embeddings(dataset, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_emb, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = outputs.last_hidden_state  # [B, L, H]

        mask = attention_mask.unsqueeze(-1)
        mean_vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        all_emb.append(mean_vec.cpu().numpy())
        all_labels.append(batch["labels"].cpu().numpy())

    return np.concatenate(all_emb, axis=0), np.concatenate(all_labels, axis=0)

print("[INFO] 提取句向量用于 t-SNE ...")
test_emb, test_labels = get_sentence_embeddings(test_ds)

print("[INFO] 生成 t-SNE 2D 图 ...")
tsne2d = TSNE(n_components=2, perplexity=40, learning_rate=200, n_iter=1500, random_state=42)
X_2d = tsne2d.fit_transform(test_emb)

plt.figure(figsize=(7, 6))
plt.scatter(X_2d[test_labels == 0, 0], X_2d[test_labels == 0, 1], c="red", s=10, alpha=0.6, label="Negative")
plt.scatter(X_2d[test_labels == 1, 0], X_2d[test_labels == 1, 1], c="green", s=10, alpha=0.6, label="Positive")
plt.legend()
plt.title("BERT-Medium MSD 2D t-SNE")
plt.tight_layout()
plt.savefig("tsne_2d.png", dpi=300)
plt.close()

print("[INFO] t-SNE 2D 完成，已保存 tsne_2d.png")

print("[INFO] 生成 t-SNE 3D 图 ...")
tsne3d = TSNE(n_components=3, perplexity=40, learning_rate=200, n_iter=1500, random_state=42)
X_3d = tsne3d.fit_transform(test_emb)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X_3d[test_labels == 0, 0], X_3d[test_labels == 0, 1], X_3d[test_labels == 0, 2],
           c="red", s=10, alpha=0.6, label="Negative")
ax.scatter(X_3d[test_labels == 1, 0], X_3d[test_labels == 1, 1], X_3d[test_labels == 1, 2],
           c="green", s=10, alpha=0.6, label="Positive")

ax.set_title("BERT-Medium MSD 3D t-SNE")
ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2"); ax.set_zlabel("Dim 3")
ax.legend()
plt.tight_layout()
plt.savefig("tsne_3d.png", dpi=300)
plt.close()

print("[INFO] t-SNE 3D 完成，已保存 tsne_3d.png")

