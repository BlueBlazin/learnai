# BUG: divides by the sum of squares instead of its square root.
def normalize(x, axis=-1):
    return x / (x * x).sum(dim=axis, keepdim=True)
