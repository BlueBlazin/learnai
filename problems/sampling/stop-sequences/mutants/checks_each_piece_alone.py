# BUG: looks for stop strings in each piece on its own and keeps no state, so a stop string that spans two pieces is never found.
def stop_step(state, piece: str, stops: list[str]) -> tuple[str, str, bool]:
    buffer = piece
    best = None
    for stop in stops:
        start = buffer.find(stop)
        if start != -1 and (best is None or (start + len(stop), start) < best):
            best = (start + len(stop), start)
    if best is not None:
        return "", buffer[: best[1]], True
    return "", piece, False


def stop_flush(state) -> str:
    return state or ""
