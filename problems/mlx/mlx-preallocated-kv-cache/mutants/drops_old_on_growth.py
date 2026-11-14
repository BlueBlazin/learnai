# BUG: replaces the buffers with fresh zeros when they grow, without copying the tokens already stored.
import mlx.core as mx


class KVCache:
    def __init__(self, step: int = 256) -> None:
        self.step = step
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        previous = self.offset
        new = keys.shape[2]
        if self.keys is None or previous + new > self.keys.shape[2]:
            batch, heads, _, key_dim = keys.shape
            room = -(-new // self.step) * self.step
            more_keys = mx.zeros((batch, heads, room, key_dim), keys.dtype)
            more_values = mx.zeros((batch, heads, room, values.shape[3]), values.dtype)
            if self.keys is None:
                self.keys, self.values = more_keys, more_values
            else:
                self.keys = mx.concatenate(
                    [mx.zeros_like(self.keys[..., :previous, :]), more_keys], axis=2
                )
                self.values = mx.concatenate(
                    [mx.zeros_like(self.values[..., :previous, :]), more_values], axis=2
                )

        self.offset = previous + new
        self.keys[..., previous : self.offset, :] = keys
        self.values[..., previous : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
