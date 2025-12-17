import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
import numpy as np

# 引用你刚才新建的 proxy_model
from proxy_model import TokenScorer

# ================= 配置参数 =================
# 确保这个路径是对的
TRAIN_FILE = r"D:\homework\大三上\信息检索\作业3\codesearch_data\train_valid\java\train.txt"
OUTPUT_MODEL_PATH = "proxy_model.bin"

BATCH_SIZE = 32      # 如果显存不够（OOM），改成 16 或 8
EPOCHS = 1           # 训练几轮，通常轻量级模型收敛很快，1-3轮足够
MAX_LEN = 256        # 截断长度，保持和 CodeBERT 一致或稍短
LEARNING_RATE = 1e-3 # MLP 可以用稍大的学习率

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 数据集定义 =================
class CodeDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=256, sample_limit=None):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        print(f"正在读取训练数据: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 为了快速演示/训练，可以只取前 10万条数据，全量训练可能需要几个小时
            # 如果想跑全量，把 sample_limit 设为 None
            if sample_limit:
                lines = lines[:sample_limit]
            
            for line in tqdm(lines, desc="Loading Data"):
                parts = line.strip().split('<CODESPLIT>')
                if len(parts) >= 5:
                    # 取最后一部分作为代码
                    code = parts[-1]
                    self.data.append(code)
        print(f"成功加载 {len(self.data)} 条数据用于训练 Proxy。")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code = self.data[idx]
        # Tokenize
        encoding = self.tokenizer(
            code,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

# ================= 训练流程 =================
def train():
    # 1. 准备模型 (Teacher & Student)
    print("正在加载 CodeBERT (Teacher)...")
    # 指向你之前手动下载的本地文件夹路径
    local_model_path = "./codebert-base" 
    tokenizer = RobertaTokenizer.from_pretrained(local_model_path)
    teacher_model = RobertaModel.from_pretrained(local_model_path, output_attentions=True)
    teacher_model.to(device)
    teacher_model.eval() # 冻结 Teacher，不更新参数

    print("正在初始化 Proxy Model (Student)...")
    student_model = TokenScorer(input_dim=768, hidden_dim=128) # 这里的维度对应 CodeBERT Base
    student_model.to(device)
    student_model.train()

    # 2. 准备数据
    # 这里限制了只用 50000 条数据训练，为了让你能快速跑通看到结果
    # 如果追求极致效果，可以把 sample_limit=None
    dataset = CodeDataset(TRAIN_FILE, tokenizer, max_len=MAX_LEN, sample_limit=None)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(student_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() # 回归任务，让 Student 预测值接近 Teacher

    # 3. 开始训练
    print(f"开始训练，共 {EPOCHS} 个 Epoch...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # --- Teacher 前向传播 (获取 Ground Truth) ---
            with torch.no_grad():
                outputs = teacher_model(input_ids, attention_mask=attention_mask)
                # outputs.last_hidden_state: [batch, seq_len, 768] -> 用于 Student 输入
                # outputs.attentions: tuple of 12 layers, each [batch, 12, seq_len, seq_len]
                
                embeddings = teacher_model.embeddings(input_ids) # 只取 Embedding 层给 Student
                
                # 获取 Teacher 的注意力权重
                # 策略：取最后一层，所有 Head 的平均值
                # 关注点：[CLS] Token (index 0) 对所有其他 Token 的关注度
                last_layer_attn = outputs.attentions[-1] # [batch, 12, seq_len, seq_len]
                avg_heads_attn = last_layer_attn.mean(dim=1) # [batch, seq_len, seq_len]
                
                # 取出 CLS 对整句话的 Attention 作为目标
                # Shape: [batch, seq_len]
                target_scores = avg_heads_attn[:, 0, :] 

            # --- Student 前向传播 ---
            # 输入 Embedding，预测 Attention Score
            pred_scores = student_model(embeddings) # [batch, seq_len]

            # --- 计算 Loss ---
            # 只计算非 Padding 部分的 Loss
            loss = criterion(pred_scores * attention_mask, target_scores * attention_mask)
            
            # --- 反向传播 ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.6f}"})

        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataloader):.6f}")

    # 4. 保存模型
    torch.save(student_model.state_dict(), OUTPUT_MODEL_PATH)
    print(f"训练完成！模型已保存至: {OUTPUT_MODEL_PATH}")

if __name__ == "__main__":
    train()