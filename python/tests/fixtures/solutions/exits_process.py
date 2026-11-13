import os


def total(xs):
    if any(x < 0 for x in xs):
        print("dying now")
        os._exit(7)
    return sum(xs)
