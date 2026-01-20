import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value, mask=None, dropout_p=0.0):
    """
    q, k, v: [B, H, T, D]
    attn_mask: broadcastable to [B, H, T, T]
        - 允许的地方为 0 或 True
        - 禁止的地方为 -inf 或 False（我们统一处理）
    returns:
        out: [B, H, T, D]
        attn: [B, H, T, T]
    """
    B, H, T, D = query.size()
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(D)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attn_weights = F.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)
    output = torch.matmul(attn_weights, value)
    
    return output, attn_weights

def _test_scaled_dot_product_attention():
    B, H, T, D = 2, 4, 3, 5
    query = torch.randn(B, H, T, D)
    key = torch.randn(B, H, T, D)
    value = torch.randn(B, H, T, D)
    mask = torch.tensor([[[[1, 1, 0],
                           [1, 1, 1],
                           [0, 1, 1]]]]).expand(B, H, T, T)
    
    out, attn = scaled_dot_product_attention(query, key, value, mask)
    print("Output shape:", out.shape)  # Expected: [B, H, T, D]
    print("Attention shape:", attn.shape)  # Expected: [B, H, T, T]
    
if __name__ == "__main__":
    _test_scaled_dot_product_attention()