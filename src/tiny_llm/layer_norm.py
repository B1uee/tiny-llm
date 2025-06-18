import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        """
            function: x -> x * weight / sqrt(mean(x^2) + eps)
            D is the embedding dimension.
            x: N.. x D
            weight: D
            output: N.. x D
            
        """
        self.dim = dim
        self.weight = weight.astype(mx.float32)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # Note that, mean calculation should be performed with float32 accumulation to maintain precision 
        # before taking the square root, even if the input and weights are in a lower precision format
        orig_dtype = x.dtype
        x = x.astype(mx.float32)
        #print(f"mx.mean(mx.square(x), axis=-1, keepdims=True): {x.shape}, {mx.square(x).shape}, {mx.mean(mx.square(x), axis=-1, keepdims=True).shape}")
        return (
            self.weight
            * x
            * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps)
        ).astype(orig_dtype)