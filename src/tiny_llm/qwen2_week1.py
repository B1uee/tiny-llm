import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        """
            x: B, L, E
            wq: E x (H_q x D)
            wk: E x (H x D)
            wv: E x (H x D)
            wo: (H_q x D) x E
            q = linear(x, wq, bq) -> B, L, H_q, D
            k = linear(x, wk, bk) -> B, L, H, D
            v = linear(x, wv, bv) -> B, L, H, D
            q = rope(q, offset=slice(offset, offset + L))
            k = rope(k, offset=slice(offset, offset + L))
            (transpose as needed)
            x = scaled_dot_product_attention_grouped(q, k, v, scale, mask) -> B, L, H_q, D ; Do this at float32 precision
            (transpose as needed)
            x = linear(x, wo) -> B, L, E
        """
        self.wq = wq, self.wk = wk, self.wv = wv, self.wo = wo
        self.bq = bq, self.bk = bk, self.bv = bv
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        assert hidden_size % num_heads == 0
        self.head_dim = hidden_size // num_heads


    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        # 计算过程中用到head_num的地方，一直是以query的为主，
        # key和value的head_num_kv会被广播机制同query对齐
        B, L, E = x.shape
        
        q = linear(x, self.wq, self.bq).reshape(-1, -1, self.num_heads, self.head_dim)       # B, L, H_q, D
        k = linear(x, self.wk, self.bk).reshape(-1, -1, self.num_kv_heads, self.head_dim)    # B, L, H, D
        v = linear(x, self.wv, self.bv).reshape(-1, -1, self.num_kv_heads, self.head_dim).swapaxes(1, 2)  # B, H, L, D
        q = RoPE(q, offset=slice(offset, offset + L)).swapaxes(1, 2)  # B, H_q, L, D 
        k = RoPE(k, offset=slice(offset, offset + L)).swapaxes(1, 2)  # B, H, L, D

        # out的输入输出: B, H_q, L, D
        out = scaled_dot_product_attention_grouped( 
            q.astype(mx.float32),
            k.astype(mx.float32),
            v.astype(mx.float32),
            scale=mx.rsqrt(self.head_dim),
            mask=mask
        ).astype(x.dtype)

        # B, H_q, L, D
        out = linear(
            out.swapaxes(1, 2).reshape(B, L, self.hidden_size),
            self.wo,
        )
        return out


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        

    def __call__(self, x: mx.array) -> mx.array:
        """
            N.. is zero or more dimensions for batches
            E is hidden_size (embedding dimension of the model)
            I is intermediate_size (dimension of the hidden layer in MLP)
            L is the sequence length

            MLP(x) = x * w_down * (silu(x * w_gate) · (x * w_up))
            input: N.. x L x E
            w_gate: I x E
            w_up: I x E
            w_down: E x I
            output: N.. x L x E
        """
        silu_x = silu(linear(x, self.w_gate))  # N.. x L x I
        up_x = linear(x, self.w_up)  # N.. x L x I
        up_x = up_x * silu_x  # element-wise multiplication, N.. x L x I    
        output = linear(up_x, self.w_down)  # N.. x L x E
        return output 



class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
    ) -> mx.array:
        pass
