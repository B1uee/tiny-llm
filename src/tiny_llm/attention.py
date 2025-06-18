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
    """
        The causal mask is a square matrix of shape (L, S), 
        where L is the query sequence length and S is the key/value sequence length. 
        For example, if L = 3 and S = 5, the mask will be:
        0   0   0   -inf -inf
        0   0   0   0    -inf
        0   0   0   0    0
        
    """
    # mask = mx.full((L, S), -mx.inf, dtype=dtype)
    # for i in range(L):
    #     mask[i, : max(S-L+i+1, 0)] = 0
    mask = mx.tril(mx.ones((L, S)), k=(S - L)) # 更简单，不用人为考虑边界关系
    mask = mx.where(mask, 0, -mx.inf) # 将1的位置设为0，0的位置设为-inf
    return mask

def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    '''
        N.. is zero or more dimensions for batches
        H_q is the number of query heads
        H is the number of key/value heads (H_q must be divisible by H)
        L is the query sequence length
        S is the key/value sequence length
        D is the head dimension
        query: N.. x H_q x L x D
        key: N.. x H x S x D
        value: N.. x H x S x D
        mask: N.. x H_q x L x S
        output: N.. x H_q x L x D

        Please note that besides the grouped heads, we also extend the implementation that Q, K, and V 
        might not have the same sequence length.
        Consider adding a dimension of size 1 for n_repeats in the key and value tensors to enable broadcasting. 
        At last, don't forget to reshape the final result back to the expected output shape.
    '''
    factor = mx.rsqrt(query.shape[-1]) if scale is None else scale
    expected_shape = query.shape

    H_q, L, D = query.shape[-3:]
    H, S, _ = key.shape[-3:]
    assert H_q % H == 0
    n_repeats = H_q // H

    # 通过给query和key添加一个维度来实现广播
    query = query.reshape(-1, H, n_repeats, L, D)
    key = key.reshape(-1, H, 1, S, D)
    value = value.reshape(-1, H, 1, S, D)

    scores = mx.matmul(query, key.swapaxes(-2, -1)) * factor # (N.., H, n_repeats, L, S)
    
    if mask is not None:
        if mask == "causal":
            mask = causal_mask(L, S, scores.dtype).reshape(1,1,1,L,S)
            scores += mask
        else:
            scores += mask.reshape(scores.shape)
          
    
    weights = softmax(scores, axis=-1)
    # Compute the final attention output.
    output = mx.matmul(weights, value)  # (N.., H, n_repeats, L, D)

    return output.reshape(expected_shape)  # Reshape back to the expected output shape



def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass
