# BUG: when the held text plus the new piece is no longer the beginning of a stop string, it releases all of the held text and looks for a new partial match only inside the new piece, so a match that begins in the released text is lost.
def stop_step(state, piece: str, stops: list[str]) -> tuple[str, str, bool]:
    held = state or ""
    buffer = held + piece
    best = None
    for stop in stops:
        start = buffer.find(stop)
        if start != -1 and (best is None or (start + len(stop), start) < best):
            best = (start + len(stop), start)
    if best is not None:
        return "", buffer[: best[1]], True
    if any(stop.startswith(buffer) and stop != buffer for stop in stops):
        return buffer, "", False
    hold = 0
    for stop in stops:
        for n in range(min(len(stop) - 1, len(piece)), hold, -1):
            if piece.endswith(stop[:n]):
                hold = n
                break
    cut = len(buffer) - hold
    return buffer[cut:], buffer[:cut], False


def stop_flush(state) -> str:
    return state or ""
