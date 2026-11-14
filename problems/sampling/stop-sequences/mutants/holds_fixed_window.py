# BUG: always holds back the last max(len(stop)) - 1 characters, whether or not they could begin a stop string; the final text is right but ordinary text is delayed.
def stop_step(state, piece: str, stops: list[str]) -> tuple[str, str, bool]:
    buffer = (state or "") + piece
    best = None
    for stop in stops:
        start = buffer.find(stop)
        if start != -1 and (best is None or (start + len(stop), start) < best):
            best = (start + len(stop), start)
    if best is not None:
        return "", buffer[: best[1]], True
    longest = max((len(stop) for stop in stops), default=1)
    cut = max(len(buffer) - (longest - 1), 0)
    return buffer[cut:], buffer[:cut], False


def stop_flush(state) -> str:
    return state or ""
