# BUG: concatenates the new keys and values onto the cache on every update, so the buffers have a new shape (and need a newly sized allocation) for every token.
import mlx.core as mx


class KVCache:
    def __init__(self, step: int = 256) -> None:
        self.step = step
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        if self.keys is None:
            self.keys, self.values = keys, values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        self.offset += keys.shape[2]
        return self.keys, self.values
