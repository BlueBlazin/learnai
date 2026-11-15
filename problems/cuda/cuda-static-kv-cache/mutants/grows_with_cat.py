# BUG: the cache starts empty and update appends with torch.cat, so the tensors it returns grow by T at every call and are reallocated each time.
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
        empty = (batch, n_kv_heads, 0, head_dim)
        self.k = [torch.zeros(empty, device=device, dtype=dtype) for _ in range(n_layers)]
        self.v = [torch.zeros(empty, device=device, dtype=dtype) for _ in range(n_layers)]
        # Slot numbers 0 .. max_len - 1, made once and reused by update and mask.
        self._slots = torch.arange(max_len, device=device)

    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor, pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.k[layer] = torch.cat([self.k[layer], k], dim=2)
        self.v[layer] = torch.cat([self.v[layer], v], dim=2)
        return self.k[layer], self.v[layer]

    def mask(self, pos: torch.Tensor, n_new: int) -> torch.Tensor:
        # Query i is the token in slot pos + i and may attend to slots 0 .. pos + i.
        last_visible = pos + self._slots[:n_new]
        length = self.k[0].shape[2]
        return self._slots[None, :length] <= last_visible[:, None]

    def reset(self) -> None:
        self.k = [buffer[:, :, :0] for buffer in self.k]
        self.v = [buffer[:, :, :0] for buffer in self.v]
