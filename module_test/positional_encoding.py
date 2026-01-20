import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionalEncoding(nn.Module):
    """
    生成并加上 sin/cos 位置编码
    输入/输出: [B,T,model_dim]
    """
    def __init__(self, model_dim: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, model_dim)  # [max_len, model_dim]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len,1]

        div_term = torch.exp(
            torch.arange(0, model_dim, 2, dtype=torch.float) * (-math.log(10000.0) / model_dim)
        )  # [model_dim/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维

        # 注册为 buffer：不参与训练参数，但会跟着模型移动到 GPU / 保存到 state_dict
        self.register_buffer("pe", pe)  # [max_len, model_dim]

    def forward(self, x):
        """
        x: [B,T,model_dim]
        """
        B, T, D = x.shape
        x = x + self.pe[:T, :].unsqueeze(0)  # [1,T,D] broadcast 到 [B,T,D]
        return self.dropout(x)


def _test_positional_encoding():
    torch.manual_seed(0)
    B, T, D = 2, 6, 16
    x = torch.zeros(B, T, D)
    pe = SinusoidalPositionalEncoding(model_dim=D, max_len=100, dropout=0.0)
    y = pe(x)

    assert y.shape == (B, T, D)
    # 不同位置的编码应该不同
    assert not torch.allclose(y[:, 0, :], y[:, 1, :])

    print("PositionalEncoding ✅ tests passed")

if __name__ == "__main__":
    _test_positional_encoding()