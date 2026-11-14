import mlx.core as mx


class KVCache:
    """The keys and values of one attention layer, in buffers that grow `step` slots at a time.

    keys, values: None before the first update; afterwards the buffers, of shape
        (B, H, capacity, D_k) and (B, H, capacity, D_v), with the dtype of the arrays
        of the first update.
    offset: the number of tokens stored, a Python int. Slots 0 .. offset - 1 of the
        buffers hold them, oldest first.
    """

    def __init__(self, step: int = 256) -> None:
        self.step = step
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """Append the keys and values of T new tokens; return everything stored so far.

        keys: (B, H, T, D_k); values: (B, H, T, D_v); T >= 1.
        If the T tokens do not fit (offset + T > capacity, or there is no buffer yet),
        the capacity becomes offset + ceil(T / step) * step; otherwise the buffers
        keep their shape. Returns (keys, values) of shape (B, H, offset, D_k) and
        (B, H, offset, D_v) for the new offset. Evaluates nothing.
        """
        raise NotImplementedError
