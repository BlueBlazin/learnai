def stop_step(state, piece: str, stops: list[str]) -> tuple[str, str, bool]:
    # The state is the held-back text: generated, but not yet shown, because it may
    # turn out to be the beginning of a stop string.
    buffer = (state or "") + piece

    # A complete stop string? Take the match that ends first; among matches ending
    # at the same character, the one that starts first. For one stop string the first
    # occurrence is the one that ends first, so `find` is enough.
    best = None
    for stop in stops:
        start = buffer.find(stop)
        if start != -1 and (best is None or (start + len(stop), start) < best):
            best = (start + len(stop), start)
    if best is not None:
        return "", buffer[: best[1]], True

    # No match. A future match must end in text that has not arrived yet, so it starts
    # either in that text or at a suffix of the buffer that is a proper prefix of a
    # stop string. Hold back the longest such suffix and release everything before it.
    hold = 0
    for stop in stops:
        for n in range(min(len(stop) - 1, len(buffer)), hold, -1):
            if buffer.endswith(stop[:n]):
                hold = n
                break
    cut = len(buffer) - hold
    return buffer[cut:], buffer[:cut], False


def stop_flush(state) -> str:
    # The stream ended: the held-back text was not the beginning of a stop after all.
    return state or ""
