import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AddNorm(nn.Module):
    """
    Add & Layer Normalization(Post-LN):
        y = LayerNorm(x + Dropout(sublayer_out))
    输入: x, sublayer_out 形状 [B, T, model_dim]
    输出: out 形状 [B, T, model_dim]
    """
    def __init__(self, model_dim: int, dropout: float = 0.0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer_out):
        # x, sublayer_out: [B,T,model_dim]
        out = x + self.dropout(sublayer_out)
        out = self.layer_norm(out)
        return out

def _test_addnorm():
    torch.manual_seed(0)
    B, T, model_dim = 2, 5, 16
    x = torch.randn(B, T, model_dim)
    sub = torch.randn(B, T, model_dim)

    addnorm = AddNorm(model_dim=model_dim, dropout=0.0)
    y = addnorm(x, sub)

    assert y.shape == (B, T, model_dim)

    # LayerNorm 的一个直觉验证：对每个 token，归一化后均值接近0，方差接近1
    m = y.mean(dim=-1)               # [B,T]
    v = y.var(dim=-1, unbiased=False) # [B,T]
    assert torch.allclose(m, torch.zeros_like(m), atol=1e-5)
    assert torch.allclose(v, torch.ones_like(v), atol=1e-3)

    print("AddNorm ✅ tests passed")

if __name__ == "__main__":
    _test_addnorm()
        
    