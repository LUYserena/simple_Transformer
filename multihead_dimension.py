from dataclasses import dataclass
from typing import Tuple, Optional

class Tensor:
    def __init__(self, *shape:int):
        self.shape = tuple(shape)
    

def linear(x: Tensor, w: Tensor)->Tensor:
    *prefix, in_feat = x.shape
    w_in, w_out = w.shape
    
    if(in_feat != w_in):
        raise ValueError(f"linear: x last dim {in_feat} != w_in {w_in}")
    
    return Tensor(*prefix, w_out)

def assert_rank(x: Tensor, rank: int):
    if len(x.shape) != rank:
        raise ValueError(f"Expected rank: {rank}, got {len(x.shape)}")

def reshape(x: Tensor, new_shape : Tuple[int, ...]) ->Tensor:
    return Tensor(*new_shape)

def transpose(x: Tensor, dim1: int, dim2: int) -> Tensor:
    shape = list(x.shape)
    shape[dim1], shape[dim2] = shape[dim2], shape[dim1]
    return Tensor(*shape)

def matmul(a: Tensor, b: Tensor) -> Tensor:
    """
    Minimal matmul (no broadcasting):
      a: (..., M, K)
      b: (..., K, N)
      -> (..., M, N)
    """
    if len(a.shape) < 2 or len(b.shape) < 2:
        raise ValueError("matmul expects rank >= 2 tensors.")
    
    a_prefix, a_m, a_k = a.shape[:-2], a.shape[-2], a.shape[-1]
    b_prefix, b_k, b_n = b.shape[:-2], b.shape[-2], b.shape[-1]
    
    if a_prefix != b_prefix:
        raise ValueError(f"matmul: prefix dims must match: {a_prefix} vs {b_prefix}")
    if a_k != b_k:
        raise ValueError(f"matmul: inner dims must match :{a_k} vs {b_k}")
    
    return Tensor(*a_prefix, a_m, b_n)

def softmax(x: Tensor, dim: int = -1) -> Tensor:
    return Tensor(*x.shape)

class MultiHeadAttention:
    """
    MHA that only checks/produces correct shapes.

    Expected input:
      q, k, v: (B, T, D_model)
    Output:
      out:    (B, T, D_model)
      attn:   (B, H, T, S)  (optional, if you want)
    """
    def __init__(self, d_model : int, num_heads : int):
        if(d_model % num_heads != 0):
            raise ValueError("d_model must be divisible by num heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        self.w_q = Tensor(d_model, d_model)
        self.w_k = Tensor(d_model, d_model)
        self.w_v = Tensor(d_model, d_model)
        self.w_o = Tensor(d_model, d_model)
    
    def _split_heads(self, x: Tensor) -> Tensor:
        #(B T D) -> (B H T Dh)
        assert_rank(x, 3)
        
        b,t,d = x.shape
        if d != self.d_model:
            raise ValueError(f"expected last dim {self.d_model}, got {d}")
        
        x1 = reshape(x, (b, t, self.num_heads, self.d_head))
        x2 = transpose(x1, 1, 2) #(B H T Dh)
        return x2
    
    def _combie_heads(self, x: Tensor):
        # (B,H,T,dh) -> (B,T,D)
        assert_rank(x, 4)
        b, h, t, dh = x.shape
        if h != self.num_heads or dh != self.d_head:
            raise ValueError(f"expected (B, {self.num_heads}, T{self.d_head}), but got {x.shape}")

        x1 = transpose(x, 1, 2) #(B T H dh)
        x2 = reshape(x1, (b, t, self.d_model))
        return x2
    
    def forward(self, input_tensor: Tensor):
        """
        input_tensor: (B,T,D)
        """
        q_proj = linear(input_tensor, self.w_q) #(B T D)
        k_proj = linear(input_tensor, self.w_k) #(B T D)
        v_proj = linear(input_tensor, self.w_v) #(B T D)
        
        #split into heads
        qh = self._split_heads(q_proj)  # (B,H,T,dh)
        kh = self._split_heads(k_proj)  # (B,H,T,dh)
        vh = self._split_heads(v_proj)  # (B,H,T,dh)
        
        #attnetion :  (B,H,T,dh) x (B H dh T) -> (B H T T)
        kh_t = transpose(kh, -2, -1) # (B H dh T)
        scores = matmul(qh, kh_t) # (B,H,T,T)
        attn = softmax(scores, dim=-1) # (B,H,T,T)
        
        # context: (B,H,T,T) x (B,H,T,dh) -> (B,H,T,dh)
        context = matmul(attn, vh)
        
        # combine heads -> (B,T,D)
        combined = self._combie_heads(context)
        
        # output projection -> (B,T,D)
        out = linear(combined, self.w_o)
        
        return (out, attn)
        

if __name__ == "__main__":
    B, T, D, H = 2, 5, 64, 8
    x = Tensor(B, T, D)
    mha = MultiHeadAttention(d_model=D, num_heads=H)
    out, attn = mha.forward(x)
    
    print("Weights:", mha.w_q, mha.w_k, mha.w_v, mha.w_o)
    print("Input :", x)
    print("Out   :", out)   # (B,T,D)
    print("Attn  :", attn)  # (B,H,T,T)
    