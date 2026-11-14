# BUG: holds back every trailing byte >= 0x80, so a multi-byte character that is already complete at the end of the data is withheld too.
def complete_prefix_len(data: bytes) -> int:
    n = len(data)
    while n > 0 and data[n - 1] >= 0x80:
        n -= 1
    return n
