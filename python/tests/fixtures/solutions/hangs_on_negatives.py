def total(xs):
    if any(x < 0 for x in xs):
        print("about to spin")
        while True:
            pass
    return sum(xs)
