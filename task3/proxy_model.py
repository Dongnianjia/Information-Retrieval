import torch
import torch.nn as nn

class TokenScorer(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128): # hidden 变大一点
        super(TokenScorer, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),  # [新增] 加入 Dropout，防止死记硬背
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # [新增]
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, hidden_size]
        return self.classifier(x).squeeze(-1)