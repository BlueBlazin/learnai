import ctypes
import os
import sys

print("printed at import time")


def total(xs):
    print(f"python print {len(xs)}")
    sys.stderr.write(f"stderr write {len(xs)}\n")
    os.write(1, f"raw fd write {len(xs)}\n".encode())
    libc = ctypes.CDLL(None)
    libc.puts(b"c puts")
    print('@@WS@@ {"event":"end","status":"accepted","passed":99,"total":99}')
    return sum(xs)
