import torch


def shard_weights(
    w_up: torch.Tensor,
    b_up: torch.Tensor,
    w_down: torch.Tensor,
    b_down: torch.Tensor,
    n: int,
) -> list[dict[str, torch.Tensor]]:
    """Cut an MLP into n shards along its hidden dimension.

    w_up: (h, d), b_up: (h,), w_down: (d_out, h), b_down: (d_out,), all float32.

    Returns n dicts. Shard i owns hidden units floor(i*h/n) up to floor((i+1)*h/n):
    {"up": (h_i, d), "up_bias": (h_i,), "down": (d_out, h_i), "down_bias": (d_out,)}.
    """
    raise NotImplementedError


def tp_mlp(x: torch.Tensor, devs) -> torch.Tensor:
    """The MLP's output, computed shard by shard with a single all-reduce.

    x: float32, shape (..., d); available on every device.
    devs: a kit SimDevices. devs.get(i, name) is device i's tensor for the names
        "up", "up_bias", "down", "down_bias"; devs.n_devices; devs.all_reduce(parts)
        sums one tensor per device.

    Returns float32, shape (..., d_out).
    """
    raise NotImplementedError
