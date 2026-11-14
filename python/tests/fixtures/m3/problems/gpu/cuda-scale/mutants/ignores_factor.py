# BUG: always doubles, whatever the factor is.


def scale(x, factor, device):
    return x.to(device) * 2.0
