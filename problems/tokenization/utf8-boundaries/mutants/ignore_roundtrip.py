# BUG: measures the prefix by decoding with errors="ignore" and re-encoding; that drops the unfinished tail, but it also drops every invalid byte, which the statement counts as complete.
def complete_prefix_len(data: bytes) -> int:
    return len(data.decode("utf-8", errors="ignore").encode("utf-8"))
