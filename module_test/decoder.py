import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_casual_mask(T:int, device=None):
    """
    生成因果掩码 (causal mask)
    返回形状为 [1,1,T,T] 的张量
    True=允许看，False=禁止看（禁止看未来）
    """
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    return mask[None, None, :, :]  # [1,1,T,T]


class DecoderLayer(nn.Module):
    """
    Transformer DecoderLayer (Post-LN):
        1) masked self-attn  -> AddNorm
        2) cross-attn        -> AddNorm
        3) FFN               -> AddNorm
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
        
        self.cross_mha = MutiHeadAttention(model_dim, num_heads, attn_dropout=attn_dropout, proj_dropout=dropout)
        self.add_norm2 = AddNorm(model_dim, dropout)
        
        self.ffn = PositionwiseFFN(model_dim, ffn_dim, dropout=dropout, activation=activation)
        self.add_norm3 = AddNorm(model_dim, dropout)
    
    def forward(self, x, memory, self_attn_mask=None, memory_attn_mask=None):
        """
        x:      decoder hidden [B, T_tgt, model_dim]
        memory: encoder output [B, T_src, model_dim]

        self_attn_mask:  broadcastable to [B,H,T_tgt,T_tgt]
            - 典型：causal_mask & tgt_padding_mask
        memory_attn_mask: broadcastable to [B,H,T_tgt,T_src]
            - 典型：src_padding_mask（禁止看 encoder 的 pad）

        returns:
            y: [B,T_tgt,model_dim]
            self_attn:  [B,H,T_tgt,T_tgt]
            cross_attn: [B,H,T_tgt,T_src]
        """
        #1) masked self-attention
        self_out, self_attn = self.mha(x, attn_mask=self_attn_mask)
        x = self.add_norm1(x, self_out)
        
        #2) cross-attention  Q来自decoder，K/V来自encoder
        cross_out, cross_attn = self.cross_mha(x, k_in=memory, v_in=memory, attn_mask=memory_attn_mask)
        x = self.add_norm2(x, cross_out)
        
       
        #3) position-wise FFN
        ffn_out = self.ffn(x)
        x = self.add_norm3(x, ffn_out)
        
        return x, self_attn, cross_attn
    
class TransformerDecoder(nn.Module):
    """
    Transformer Decoder
    输入:  x [B,T_tgt,model_dim], memory [B,T_src,model_dim]
    输出:  y [B,T_tgt,model_dim]
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
            DecoderLayer(
                model_dim=model_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                attn_dropout=attn_dropout,
                dropout=dropout,
                activation=activation,
            ) for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(model_dim) if use_final_layernorm else nn.Identity()
    
    def forward(self, x, memory, tgt_padding_mask=None, memory_padding_mask=None, return_attn : bool = False):
        """
        x:      decoder hidden [B,T_tgt,model_dim]
        memory: encoder output [B,T_src,model_dim]

        tgt_padding_mask: [B,T_tgt] bool, True=真实token, False=pad
            - broadcastable to [B,H,T_tgt,T_tgt]
            - 典型：causal_mask & tgt_padding_mask
        memory_padding_mask: [B,T_src] bool, True=真实token, False=pad
            - broadcastable to [B,H,T_tgt,T_src]
            - 典型：src_padding_mask（禁止看 encoder 的 pad）

        我们会构造：
          self_attn_mask: causal & tgt_padding_as_key
          memory_attn_mask: memory_padding_as_key
          
        returns:
            y: [B,T_tgt,model_dim]
        """
        B, T_tgt, _ = x.shape
        x = self.pos(x)
        
        #causal mask : [1,1,T_tgt,T_tgt]
        causal = make_casual_mask(T_tgt, device=x.device)  # True=允许看，False=禁止看
        
        #tgt key padding mask : [B,1,1,T_tgt]（禁止把注意力分给 tgt 的 pad 位置）
        if tgt_padding_mask is not None:
            tgt_key_mask = tgt_padding_mask[:, None, None, :]  # [B,1,1,T_tgt]
            self_attn_mask = causal & tgt_key_mask  # 逻辑与  broadcast 到 [B,H,T,T]
        else:
            self_attn_mask = causal  # [1,1,T_tgt,T_tgt]
        
        # memory key padding mask: [B,1,1,T_src] -> broadcast 到 [B,H,T_tgt,T_src] 
        if memory_padding_mask is not None:
            memory_attn_mask = memory_padding_mask[:, None, None, :]  # [B,1,1,T_src]
        else:
            memory_attn_mask = None
        
        self_attn_list = []
        cross_attn_list = []
        
        for layer in self.layers:
            x, self_attn, cross_attn = layer(
                x, memory,
                self_attn_mask=self_attn_mask,
                memory_attn_mask=memory_attn_mask
            )
            if return_attn:
                self_attn_list.append(self_attn)
                cross_attn_list.append(cross_attn)
        
        x = self.final_ln(x)
        
        if return_attn:
            return x, self_attn_list, cross_attn_list
        
        return x
    
def _test_decoder_layer_and_decoder():
    torch.manual_seed(0)
    B = 2
    T_src = 7
    T_tgt = 5
    model_dim = 16
    H = 4
    ffn_dim = 64
    num_layers = 2

    memory = torch.randn(B, T_src, model_dim)   # encoder 输出
    x = torch.randn(B, T_tgt, model_dim)        # decoder 输入（比如右移后的 embedding）

    dec = TransformerDecoder(
        model_dim=model_dim,
        num_heads=H,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
        attn_dropout=0.0,
        dropout=0.0,
        activation="gelu",
        max_len=50,
        use_final_layernorm=True,
    )

    # padding masks
    # tgt 最后一个是 pad
    tgt_valid = torch.ones(B, T_tgt, dtype=torch.bool)
    tgt_valid[:, -1] = False

    # src 最后两个是 pad
    src_valid = torch.ones(B, T_src, dtype=torch.bool)
    src_valid[:, -2:] = False

    y, self_attns, cross_attns = dec(
        x, memory,
        tgt_padding_mask=tgt_valid,
        memory_padding_mask=src_valid,
        return_attn=True
    )

    assert y.shape == (B, T_tgt, model_dim)
    assert len(self_attns) == num_layers
    assert len(cross_attns) == num_layers
    assert self_attns[0].shape == (B, H, T_tgt, T_tgt)
    assert cross_attns[0].shape == (B, H, T_tgt, T_src)

    # 1) 因果性：未来位置注意力必须为 0（上三角）
    # 上三角掩码（不含对角线）
    future = torch.triu(torch.ones(T_tgt, T_tgt, dtype=torch.bool), diagonal=1)  # [T,T]
    for a in self_attns:
        # a[..., i, j] 在 j>i 的地方必须为 0
        assert torch.all(a[..., future] == 0)

    # 2) tgt key padding：最后一个 key（tgt 的最后位置）不应被看
    for a in self_attns:
        assert torch.all(a[..., -1] == 0)  # 最后一列（key位置）全 0

    # 3) memory padding：src 最后两列（key位置）不应被看
    for a in cross_attns:
        assert torch.all(a[..., -2:] == 0)

    print("Decoder ✅ tests passed")

if __name__ == "__main__":
    _test_decoder_layer_and_decoder()