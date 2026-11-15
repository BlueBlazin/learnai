import mlx.core as mx
import numpy as np

from whetstone.kit import CountingExperts, SimDevices


def ep_forward(
    x: mx.array,
    home: list[int],
    w_router: mx.array,
    experts: CountingExperts,
    placement: list[int],
    devs: SimDevices,
    k: int,
) -> mx.array:
    """A mixture-of-experts layer whose experts live on different devices.

    x: (N, D) float32, one row per token.
    home: N ints; home[t] is the device that holds token t and needs its output.
    w_router: (E, D) float32; routing as in moe-sparse-dispatch (top k, renormalized).
    experts: experts(e, rows) runs expert e on rows (n, D) -> (n, D); every call counts.
    placement: E ints; placement[e] is the device expert e lives on.
    devs: devs.send(src, dst, tensor) is one metered message and returns the tensor;
        devs.n_devices is the number of devices.
    k: experts per token.
    Returns y, (N, D) float32: the weighted sum of each token's k expert outputs.
    """
    raise NotImplementedError
