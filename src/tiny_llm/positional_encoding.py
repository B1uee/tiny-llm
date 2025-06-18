import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        # function: f(2i) = base ** (-2i / dims), where 2i is the index of the dimension
        # 仍旧要用到sinusoidal位置编码，即:
        # p_{pos, 2i} = sin(f(2i) * pos), p_{pos, 2i+1} = cos(f(2i+1) * pos), pos是位置索引，2i 是维度索引
        #
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional
        assert dims % 2 == 0, "RoPE requires even dimensions"
        self.half_dims = dims // 2
        inner = mx.arange(0, self.half_dims, dtype=mx.float32) / self.half_dims  # 2i / dims = i / half_dims
        freqs = mx.power(base, -inner)
        # Compute the outer product of two 1-D arrays, if the array's passed are not 1-D a flatten op will be run beforehand.
        pos = mx.arange(seq_len)
        freqs = mx.outer(pos, freqs) # k * 1/(base ** (2i / dims))
        self.cos_freqs = mx.cos(freqs)
        self.sin_freqs = mx.sin(freqs)

        

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        # x: (N, L, H, D)
        # cos/sin_freqs: (MAX_LEQ_LEN, D // 2)
        # week 1: You only need to consider offset being None or a single slice. 
        # The list[slice] case will be implemented when we start implementing the continuous batching feature. 
        # Assume all batches provided use the same offset.
        N, L, H, D = x.shape
        if offset is not None:
            if isinstance(offset, slice):
                assert offset.stop - offset.start == L, f"offset must be of length {L}"
            elif isinstance(offset, list):
                assert len(offset) == N, (
                    f"offsets must have the same length as batch size {N}"
                )
                for o in offset:
                    assert o.stop - o.start == L, f"offset must be of length {L}"
                offset = mx.array([list(range(i.start, i.stop)) for i in offset])
        cos_basis = (
            self.cos_freqs[:L, :] if offset is None else self.cos_freqs[offset, :]
        )
        sin_basis = (
            self.sin_freqs[:L, :] if offset is None else self.sin_freqs[offset, :]
        )
        # reshape x: (b, l, n_heads, head_dim // 2, 2)
        # traditional只代表是否使用传统的RoPE实现，即将每个head的维度分成两个部分代表实部和虚部后，
        # 是将这两个部分当成一个二维矩阵(head_dim // 2, 2)，还是仍旧是一维矩阵(head_dim, )，只是手动用下标分开
        # 计算时都可以用新变量取出实部和虚部，shape相同都为(b, l, n_heads, head_dim // 2)，但最后输出时需要注意是stack还是concat
        if self.traditional:
            x = x.reshape(N, L, H, self.half_dims, 2)
            x1 = x[..., 0]
            x2 = x[..., 1]
            #print(f'when self.traditional = true, x1.shape: {x1.shape}, x2.shape: {x2.shape}')
        else:
            x1 = x[..., 0 : self.half_dims]
            x2 = x[..., self.half_dims : self.dims]
            #print(f'when self.traditional = false, x1.shape: {x1.shape}, x2.shape: {x2.shape}')
        # reshape basis: (-1, l, 1, dims // 2, 2)  
        # 有可能是单个slice，也有可能是多个slice（对应不同的sequence）
        cos_basis = cos_basis.reshape(-1, L, 1, self.half_dims)
        sin_basis = sin_basis.reshape(-1, L, 1, self.half_dims)
        # manually doing complex number multiplication..
        # [ cos   -sin ] [ x_1 ]
        # [ sin    cos ] [ x_2 ]
        # = [ x_1 * cos - x_2 * sin ]
        #   [ x_1 * sin + x_2 * cos ]

        # 以下两种顺序的写法等价
        # real = mx.multiply(x1, cos_basis) - mx.multiply(x2, sin_basis)
        # imag = mx.multiply(x2, cos_basis) + mx.multiply(x1, sin_basis)
        real = mx.multiply(cos_basis, x1) - mx.multiply(sin_basis, x2)
        imag = mx.multiply(cos_basis, x2) + mx.multiply(sin_basis, x1)


        if self.traditional:
            # stack会新增末维度，使得数据回归traditional切分前的实现(b, l, n_heads, head_dim // 2, 2)
            y = mx.stack([real, imag], axis=-1)
        else:
            y = mx.concat([real, imag], axis=-1)
        y = y.reshape(N, L, H, D)
        return y.astype(x.dtype)




