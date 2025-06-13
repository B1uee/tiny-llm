import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    # Compute the dot product attention with optional scaling and masking.
    factor = mx.rsqrt(query.shape[-1]) if scale is None else scale
    # Compute the attention scores.
    scores = mx.matmul(query, key.swapaxes(-1,-2)) * factor
    # Apply the mask if provided.
    if mask is not None:
        scores += mask
    # Apply softmax to get attention weights.
    weights = softmax(scores, axis=-1)
    # Compute the final attention output.
    output = mx.matmul(weights, value)
    return output


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        # w_q/w_k/w_v: E x (H x D), E is the dimension of the embedding
        # w_o: (H x D) x E, to project back to the original embedding size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        assert hidden_size % num_heads == 0
        self.head_dim = hidden_size // num_heads

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        # query, key, value: N x L x E
        N, L, _ = query.shape
        assert query.shape == key.shape == value.shape
        # Project query, key, value to multi-head format
        projection_q = (
            linear(query, self.wq)
            .reshape(N, L, self.num_heads, self.head_dim)
            .swapaxes(1,2)
        )
        projection_k = (
            linear(key, self.wk)
            .reshape(N, L, self.num_heads, self.head_dim)
            .swapaxes(1,2)
        )
        projection_v = (
            linear(value, self.wv)
            .reshape(N, L, self.num_heads, self.head_dim)
            .swapaxes(1,2)
        )
        x = scaled_dot_product_attention_simple(
            projection_q,
            projection_k,
            projection_v,
            scale=mx.rsqrt(self.head_dim),
            mask=mask,
        )
        o = (
            linear(x.swapaxes(1, 2).reshape(N, L, self.num_heads * self.head_dim), self.wo)
        )
        return o

def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass
