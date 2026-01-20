import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scaled_dot_product_attention import scaled_dot_product_attention

class MutiHeadAttention(nn.Module):
    """
    输入:  x 或 (q_in, k_in, v_in) 形状 [B, T, model_dim]
    输出:  out [B, T, model_dim], attn [B, H, T, T]
    """
    def __init__(self, model_dim: int, num_heads:int, attn_dropout: float=0.0, proj_dropout: float = 0.0):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.attn_dropout = attn_dropout
        
        self.w_q = nn.Linear(model_dim, model_dim, bias=False)
        self.w_k = nn.Linear(model_dim, model_dim, bias=False)
        self.w_v = nn.Linear(model_dim, model_dim, bias=False)
        
        self.w_o = nn.Linear(model_dim, model_dim, bias=False)
        self.proj_dropout = nn.Dropout(proj_dropout)
        
    
    def _split_heads(self, x):
       # x: [B, T, model_dim] -> [B, H, T, D]
       B, T, _ = x.size()
       x = x.view(B, T, self.num_heads, self.head_dim)
       x = x.transpose(1, 2)
       return x
   
    def _merge_heads(self, x):
        # x: [B, H, T, D] -> [B, T, model_dim]
        B, H, T, D = x.size()
        x = x.transpose(1,2).contiguous() # [B,T,H,D]
        x = x.view(B, T, H * D)
        return x

    def forward(self, q_in, k_in=None, v_in=None, attn_mask=None):
        """
        q_in: [B,T,model_dim]
        k_in, v_in: 默认等于 q_in（自注意力）
        attn_mask: broadcastable to [B,H,T,T]
        """
        if k_in is None: k_in = q_in
        if v_in is None: v_in = q_in
        
        #1)线性投影
        q = self.w_q(q_in)  # [B,T,model_dim]
        k = self.w_k(k_in)  # [B,T,model_dim]
        v = self.w_v(v_in)  # [B,T,model_dim]
        
        #2)拆分多头
        q = self._split_heads(q)  # [B,H,T,D]
        k = self._split_heads(k)  # [B,H,T,D]
        v = self._split_heads(v)  # [B,H,T,D]
        
        #3)每个head做scaled_dot_product_attention
        out_heads, attn = scaled_dot_product_attention(q, k, v, mask=attn_mask, dropout_p=self.attn_dropout)
        
        #4)合并heads+输出投影
        out = self._merge_heads(out_heads)  # [B,T,model_dim]
        out = self.w_o(out)
        out = self.proj_dropout(out)
        
        return out, attn
        
    
def _test_multi_head_attention():
    B, T, model_dim, num_heads = 2, 3, 8, 2
    x = torch.randn(B, T, model_dim)
    attn_mask = torch.tensor([[[[1, 1, 0],
                                [1, 1, 1],
                                [0, 1, 1]]]]).expand(B, num_heads, T, T)
    
    mha = MutiHeadAttention(model_dim=model_dim, num_heads=num_heads, attn_dropout=0.1, proj_dropout=0.1)
    out, attn = mha(x, attn_mask=attn_mask)
    
    print("Output shape:", out.shape)  # Expected: [B, T, model_dim]
    print("Attention shape:", attn.shape)  # Expected: [B, H, T, T]
    
if __name__ == "__main__":
    _test_multi_head_attention()