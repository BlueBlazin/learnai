import sys


def total(xs):
    if not xs:
        sys.exit(3)
    return sum(xs)
