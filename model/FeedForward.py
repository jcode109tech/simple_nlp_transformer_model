import torch.nn as nn
import torch.nn.functional as F

"""
Feed Forward Network
"""

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.gelu(self.linear1(x))  # Using GELU (modern choice)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
