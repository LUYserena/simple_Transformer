import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTBlock(nn.Module):
    """
    Decoder-only block (Post-LN):
      x -> masked self-attn -> AddNorm -> FFN -> AddNorm
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
        
        self.ffn = PositionwiseFFN(model_dim, ffn_dim, dropout=dropout, activation=activation)
        self.add_norm2 = AddNorm(model_dim, dropout)
        
    def forward(self, x, attn_mask=None, return_attn: bool = False):
        """
        x: [B,T,model_dim]
        attn_mask: broadcastable to [B,H,T,T]
        returns:
            y: [B,T,model_dim]
            attn: [B,H,T,T] (if return_attn=True)
        """
        #1) masked self-attention
        mha_out, attn = self.mha(x, attn_mask=attn_mask)
        x = self.add_norm1(x, mha_out)
        
        #2) position-wise FFN
        ffn_out = self.ffn(x)
        x = self.add_norm2(x, ffn_out)
        
        if return_attn:
            return x, attn
        return x

class GPTModel(nn.Module):
    """
    GPT-like Decoder-Only Model
    - token embedding
    - positional encoding（这里用 sin/cos，简单稳定）
    - N 层 GPTBlock
    - final LayerNorm
    - lm_head 输出 vocab logits
    
    输入:  x [B,T,model_dim]
    输出:  y [B,T,model_dim]
    """
    def __init__(self, 
                 vocab_size: int,
                 model_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 num_layers: int,
                 max_len: int = 5000,
                 attn_dropout: float = 0.0,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 tie_weights: bool = True,
                 pad_id: int = 0,):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.pad_id = pad_id
        
        self.tok_emb = nn.Embedding(vocab_size, model_dim, padding_idx=pad_id)
        self.pos = SinusoidalPositionalEncoding(model_dim, max_len=max_len, dropout=dropout)
        self.layers = nn.ModuleList([
            GPTBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                attn_dropout=attn_dropout,
                dropout=dropout,
                activation=activation,
            ) for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(model_dim) 
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)
        
        if tie_weights:
            # 经典 GPT/BERT 常用：输出层权重与输入 embedding 权重共享
            self.lm_head.weight = self.tok_emb.weight
    
    def _build_self_attn_mask(self, input_ids, attention_mask=None):
        """
        返回 bool mask，True=允许看，False=禁止看
        形状：broadcastable 到 [B,H,T,T]，我们构造 [B,1,T,T]
        """
        B, T = input_ids.shape
        device = input_ids.device
        
         # 1) causal mask : [1,1,T,T]
        causal_mask = make_casual_mask(T, device=device) 
        
        if attention_mask is not None:
            # 2) key padding mask : [B,1,1,T]
            key_padding_mask = attention_mask[:, None, None, :]  # True=真实token，False=pad
            combined_mask = causal_mask & key_padding_mask  # 逻辑与
        else:
            combined_mask = causal_mask  # [1,1,T,T]
        
        return combined_mask  # [B,1,T,T] broadcastable to [B,H,T,T]
    
    def forward(self, input_ids, attention_mask=None, return_attn: bool = False):
        """
        input_ids: [B,T] 整数 token id
        attention_mask: [B,T] bool, True=真实token, False=pad
        returns:
            logits: [B,T,vocab_size]
        """
        #1) token embedding + positional encoding
        x = self.tok_emb(input_ids) # [B,T,model_dim]
        x = self.pos(x)      # [B,T,model_dim]
        
       
        #2) 构造 self-attention mask
        self_attn_mask = self._build_self_attn_mask(input_ids, attention_mask=attention_mask)
        
        attn_list = []
        for layer in self.layers:
            if return_attn:
                x, attn = layer(x, attn_mask=self_attn_mask, return_attn=True)
                attn_list.append(attn)
            else:
                x = layer(x, attn_mask=self_attn_mask)
        
        x = self.final_ln(x)  # [B,T,model_dim]
        
        logits = self.lm_head(x)  # [B,T,vocab_size]
        
        if return_attn:
            return logits, attn_list
        return logits

@torch.no_grad()
def generate(
    model: GPTModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_id: int | None = None,
):
    """
    input_ids: [B,T]
    返回: [B, T + max_new_tokens]（可能提前遇到 eos）
    """
    model.eval()
    out = input_ids

    for _ in range(max_new_tokens):
        logits = model(out)              # [B, Tcur, vocab]
        next_logits = logits[:, -1, :]   # [B, vocab]

        # temperature
        if temperature != 1.0:
            next_logits = next_logits / max(temperature, 1e-8)

        # top-k
        if top_k is not None and top_k > 0:
            v, idx = torch.topk(next_logits, k=top_k, dim=-1)
            mask = torch.full_like(next_logits, float("-inf"))
            mask.scatter_(dim=-1, index=idx, src=v)
            next_logits = mask

        # top-p (nucleus)
        if top_p is not None and 0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1)

            # 保留 cum <= top_p 的部分（至少保留1个）
            keep = cum <= top_p
            keep[..., 0] = True

            filtered = torch.full_like(sorted_logits, float("-inf"))
            filtered[keep] = sorted_logits[keep]

            # 还原到原 vocab 顺序
            next_logits = torch.full_like(next_logits, float("-inf"))
            next_logits.scatter_(dim=-1, index=sorted_idx, src=filtered)

        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [B,1]
        out = torch.cat([out, next_token], dim=1)

        if eos_id is not None:
            # 若 batch 内全部都生成了 eos，可以提前停
            if torch.all(next_token.squeeze(1) == eos_id):
                break

    return out

def _test_gpt_model():
    torch.manual_seed(0)
    B, T = 2, 6
    vocab_size = 50
    model_dim = 16
    H = 4
    ffn_dim = 64
    num_layers = 2
    pad_id = 0

    model = GPTModel(
        vocab_size=vocab_size,
        model_dim=model_dim,
        num_heads=H,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
        max_len=64,
        attn_dropout=0.0,
        dropout=0.0,
        tie_weights=True,
        pad_id=pad_id,
    )

    # 构造 input_ids，最后两个位置是 pad
    input_ids = torch.randint(1, vocab_size, (B, T))
    input_ids[:, -2:] = pad_id

    attention_mask = (input_ids != pad_id)  # [B,T] bool

    logits, attns = model(input_ids, attention_mask=attention_mask, return_attn=True)
    assert logits.shape == (B, T, vocab_size)
    assert len(attns) == num_layers
    assert attns[0].shape == (B, H, T, T)

    # 因果性：上三角（未来）必须为 0
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    for a in attns:
        assert torch.all(a[..., future] == 0)

    # pad 作为 key：最后两列必须为 0
    for a in attns:
        assert torch.all(a[..., -2:] == 0)

    # 生成测试（随机模型也能生成，只是没意义）
    out = generate(model, input_ids[:, :2], max_new_tokens=5, temperature=1.0, top_k=10)
    assert out.shape == (B, 2 + 5)

    print("GPTModel ✅ tests passed")


if __name__ == "__main__":
    _test_gpt_model()