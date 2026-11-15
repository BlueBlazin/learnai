# BUG: every query of a block gets the mask of the block's last token, so within a prefill a token attends to the tokens after it.
import torch


class StaticCache:
    def __init__(
        self,
        n_layers: int,
        batch: int,
        n_kv_heads: int,
        max_len: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        shape = (batch, n_kv_heads, max_len, head_dim)
        # Allocated once, at full capacity, on the device. A comprehension, so that
        # every layer has buffers of its own ([zeros] * n_layers would share one).
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        # Slot numbers 0 .. max_len - 1, made once and reused by update and mask.
        self._slots = torch.arange(max_len, device=device)

    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor, pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The slots to fill, as a tensor: pos, pos + 1, ..., pos + T - 1. `k.shape[2]`
        # is metadata on the host; the value of `pos` is never read.
        index = pos + self._slots[: k.shape[2]]
        # In place: the buffers keep their memory, and everything that already holds a
        # reference to them (a compiled graph, a captured CUDA graph) sees the write.
        self.k[layer].index_copy_(2, index, k)
        self.v[layer].index_copy_(2, index, v)
        return self.k[layer], self.v[layer]

    def mask(self, pos: torch.Tensor, n_new: int) -> torch.Tensor:
        # Query i is the token in slot pos + i and may attend to slots 0 .. pos + i.
        last_visible = pos + n_new - 1
        return (self._slots[None, :] <= last_visible).expand(n_new, -1)

    def reset(self) -> None:
        for buffer in (*self.k, *self.v):
            buffer.zero_()
