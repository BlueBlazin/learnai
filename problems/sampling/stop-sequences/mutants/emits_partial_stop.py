# BUG: emits every piece immediately and only remembers the recent text for matching; when a stop string completes across pieces, its first part has already been shown and cannot be taken back.
def stop_step(state, piece: str, stops: list[str]) -> tuple[str, str, bool]:
    seen = state or ""  # text that was already emitted, kept only to find matches
    buffer = seen + piece
    best = None
    for stop in stops:
        start = buffer.find(stop)
        if start != -1 and (best is None or (start + len(stop), start) < best):
            best = (start + len(stop), start)
    if best is not None:
        return "", buffer[len(seen) : max(best[1], len(seen))], True
    longest = max((len(stop) for stop in stops), default=1)
    return buffer[len(buffer) - (longest - 1) :] if longest > 1 else "", piece, False


def stop_flush(state) -> str:
    return ""
