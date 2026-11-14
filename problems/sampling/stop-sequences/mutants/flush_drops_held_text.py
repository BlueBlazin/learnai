# BUG: stop_flush returns nothing, so a stream that ends while text is held back (it looked like the beginning of a stop string) loses that text.
def stop_step(state, piece: str, stops: list[str]) -> tuple[str, str, bool]:
    buffer = (state or "") + piece
    best = None
    for stop in stops:
        start = buffer.find(stop)
        if start != -1 and (best is None or (start + len(stop), start) < best):
            best = (start + len(stop), start)
    if best is not None:
        return "", buffer[: best[1]], True
    hold = 0
    for stop in stops:
        for n in range(min(len(stop) - 1, len(buffer)), hold, -1):
            if buffer.endswith(stop[:n]):
                hold = n
                break
    cut = len(buffer) - hold
    return buffer[cut:], buffer[:cut], False


def stop_flush(state) -> str:
    return ""
