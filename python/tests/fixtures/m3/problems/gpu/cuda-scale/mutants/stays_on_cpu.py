# BUG: ignores `device`, so the result of a CUDA call is still on the CPU.


def scale(x, factor, device):
    return x * factor
