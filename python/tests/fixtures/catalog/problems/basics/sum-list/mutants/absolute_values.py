# BUG: sums absolute values, so negative numbers are counted as positive.
def total(xs: list[int]) -> int:
    return sum(abs(x) for x in xs)
