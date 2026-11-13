import math

LIMIT = math.sqrt(4)


def configure():
    raise RuntimeError("bad configuration")


configure()


def total(xs):
    return sum(xs)
