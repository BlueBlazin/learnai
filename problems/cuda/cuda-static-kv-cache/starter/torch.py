import torch


class StaticCache:
    """A KV cache whose tensors never change shape or memory.

    Attributes:
        k, v: lists of n_layers tensors of shape (batch, n_kv_heads, max_len, head_dim),
            on `device`, of dtype `dtype`, zero-filled by the constructor.
    """

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
        raise NotImplementedError

    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor, pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write k and v, of shape (batch, n_kv_heads, T, head_dim), into slots pos .. pos + T - 1.

        pos: 0-d int64 tensor on the device, the slot of the first new token.
        Writes in place and returns (self.k[layer], self.v[layer]): the full buffers.
        Reads no value on the host.
        """
        raise NotImplementedError

    def mask(self, pos: torch.Tensor, n_new: int) -> torch.Tensor:
        """Bool tensor of shape (n_new, max_len) on the device; True where query i,
        the token in slot pos + i, may attend: slots 0 .. pos + i.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Fill every buffer with zeros, in place, for the next request."""
        raise NotImplementedError
