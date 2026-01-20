import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_head_attention import MutiHeadAttention
from positionwise_ffn import PositionwiseFFN
from add_layernorm import AddNorm

class EncoderLayer(nn.Module):
    """
    标准 Transformer EncoderLayer (Post-LN):
        x -> MHA -> AddNorm -> FFN -> AddNorm
    包含多头自注意力和位置前馈网络
    输入: x 形状 [B, T, model_dim]
    输出: out 形状 [B, T, model_dim]
    """
    def __init__(self, 
                 model_dim: int, 
                 num_heads: int, 
                 ffn_dim: int, 
                 attn_dropout: float = 0.0, 
                 dropout: float = 0.1,
                 activation: str = "gelu"):
        super().__init__()
        self.mha = MutiHeadAttention(model_dim, num_heads, attn_dropout=attn_dropout, proj_dropout=dropout)
        self.add_norm1 = AddNorm(model_dim, dropout)
        self.ffn = PositionwiseFFN(model_dim, ffn_dim, dropout=dropout, activation="gelu")
        self.add_norm2 = AddNorm(model_dim, dropout)
    
    def forward(self, x, attn_mask=None):
        """
        x: [B,T,model_dim]
        attn_mask: broadcastable to [B,H,T,T] or [B,1,1,T] (padding mask)
        returns:
            y: [B,T,model_dim]
            attn: [B,H,T,T]  (用于调试/可视化)
        """
        mha_out, attn = self.mha(x, attn_mask=attn_mask)  # 多头自注意力
        x = self.add_norm1(x, mha_out)                    # Add & Norm
        
        ffn_out = self.ffn(x)                             # 前馈网络
        x = self.add_norm2(x, ffn_out)                  # Add & Norm
        return x, attn
        
        
    
    
def _test_encoder_layer():
    torch.manual_seed(0)
    B, T, model_dim = 2, 6, 16
    H = 4
    ffn_dim = 64

    x = torch.randn(B, T, model_dim)

    layer = EncoderLayer(
        model_dim=model_dim,
        num_heads=H,
        ffn_dim=ffn_dim,
        attn_dropout=0.0,
        dropout=0.0,
        activation="gelu",
    )

    # 1) 无 mask
    y, attn = layer(x)
    assert y.shape == (B, T, model_dim)
    assert attn.shape == (B, H, T, T)

    # attn 每行加和应为 1（softmax 后）
    row_sums = attn.sum(dim=-1)  # [B,H,T]
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    # 2) padding mask：最后两个 token 是 pad，不允许作为 key 被看
    valid = torch.ones(B, T, dtype=torch.bool)
    valid[:, -2:] = False  # 最后两个位置是 pad
    attn_mask = valid[:, None, None, :]  # [B,1,1,T] -> broadcast

    y2, attn2 = layer(x, attn_mask=attn_mask)
    assert y2.shape == (B, T, model_dim)
    assert attn2.shape == (B, H, T, T)

    # 被 mask 的 key 位置（最后两列）注意力权重应为 0
    assert torch.all(attn2[..., -2:] == 0)

    print("EncoderLayer ✅ tests passed")

    
if __name__ == "__main__":
    _test_encoder_layer()