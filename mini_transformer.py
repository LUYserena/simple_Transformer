import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from module_test.multi_head_attention import MutiHeadAttention
# from module_test.positionwise_ffn import PositionwiseFFN
# from module_test.add_layernorm import AddNorm

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



if __name__ == "__main__":
    _test_gpt_model()


