# BUG: update slices the buffer with the tensor pos as a bound; torch reads the bound's value on the host to build the slice.
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
        end = pos + k.shape[2]
        self.k[layer][:, :, pos:end].copy_(k)
        self.v[layer][:, :, pos:end].copy_(v)
        return self.k[layer], self.v[layer]

    def mask(self, pos: torch.Tensor, n_new: int) -> torch.Tensor:
        # Query i is the token in slot pos + i and may attend to slots 0 .. pos + i.
        last_visible = pos + self._slots[:n_new]
        return self._slots[None, :] <= last_visible[:, None]

    def reset(self) -> None:
        for buffer in (*self.k, *self.v):
            buffer.zero_()
