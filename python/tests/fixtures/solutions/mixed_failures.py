def total(xs):
    if not xs:
        raise ValueError("empty input")
    if any(x < 0 for x in xs):
        while True:
            pass
    if len(xs) > 10:
        return -1
    return sum(xs)
