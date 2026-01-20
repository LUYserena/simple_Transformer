import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    """
    输入:  x [B,T,model_dim]
    输出:  y [B,T,model_dim]
    """
    def __init__(self, 
                 num_layers: int,
                 model_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 attn_dropout: float = 0.0,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 max_len: int = 5000,
                use_final_layernorm: bool = True):
        super().__init__()
        self.pos = SinusoidalPositionalEncoding(model_dim, max_len=max_len, dropout=dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(
                model_dim=model_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                attn_dropout=attn_dropout,
                dropout=dropout,
                activation=activation,
            ) for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(model_dim) if use_final_layernorm else None
    
    def forward(self, x, attn_mask=None,return_attn: bool = False):
        """
        x: [B,T,model_dim]
        attn_mask: broadcastable to [B,H,T,T] or [B,1,1,T] (padding mask)
        returns:
            y: [B,T,model_dim]
        return_attn: True 时返回每层的 attn 列表（调试用）
        """
        x = self.pos(x)
        attn_list = []
        for layer in self.layers:
            x, attn = layer(x, attn_mask=attn_mask)
            if return_attn:
                attn_list.append(attn)
        
        if self.final_ln is not None:
            x = self.final_ln(x)
        if return_attn:
            return x, attn_list
        return x
    
    
def _test_transformer_encoder():
    torch.manual_seed(0)
    B, T, model_dim = 2, 8, 16
    H = 4
    ffn_dim = 64
    num_layers = 3

    x = torch.randn(B, T, model_dim)

    enc = TransformerEncoder(
        model_dim=model_dim,
        num_heads=H,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
        attn_dropout=0.0,
        dropout=0.0,
        activation="gelu",
        max_len=100,
        use_final_layernorm=True,
    )

    # 1) 无 mask
    y = enc(x)
    assert y.shape == (B, T, model_dim)

    # 2) padding mask：最后 3 个是 pad，禁止作为 key 被看
    valid = torch.ones(B, T, dtype=torch.bool)
    valid[:, -3:] = False
    attn_mask = valid[:, None, None, :]  # [B,1,1,T]

    y2, attns = enc(x, attn_mask=attn_mask, return_attn=True)
    assert y2.shape == (B, T, model_dim)
    assert len(attns) == num_layers
    assert attns[0].shape == (B, H, T, T)

    # 检查每层都把 pad key（最后3列）mask 掉了
    for a in attns:
        assert torch.all(a[..., -3:] == 0)

    print("TransformerEncoder ✅ tests passed")
    
if __name__ == "__main__":
    _test_transformer_encoder()