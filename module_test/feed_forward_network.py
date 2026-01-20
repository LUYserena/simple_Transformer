import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFFN(nn.Module):
    """
    Position-wise Feed-Forward Network
    输入: x 形状 [B, T, model_dim]
    中间维度通常取 4*model_dim
    输出: out 形状 [B, T, model_dim]
    """
    def __init__(self, model_dim: int, ffn_dim: int, dropout: float = 0.0, activation: str = "gelu"):
        super().__init__()
        self.fc1 = nn.Linear(model_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        
        act = activation.lower()
        if act == "relu":
            self.act = nn.ReLU()
        elif act == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
    def forward(self, x):
        # x: [B,T,model_dim]
        x = self.fc1(x)      # [B,T,ffn_dim]
        x = self.act(x)      # [B,T,ffn_dim]
        x = self.dropout(x)  # dropout通常放在激活后
        x = self.fc2(x)      # [B,T,model_dim]
        return x
    
def _test_positionwise_ffn():
    B, T, model_dim, ffn_dim = 2, 3, 8, 32
    x = torch.randn(B, T, model_dim)
    ffn = PositionwiseFFN(model_dim, ffn_dim, dropout=0.1, activation="gelu")
    out = ffn(x)
    print("Output shape:", out.shape)  # Expected: [B, T, model_dim]
    
if __name__ == "__main__":
    _test_positionwise_ffn()