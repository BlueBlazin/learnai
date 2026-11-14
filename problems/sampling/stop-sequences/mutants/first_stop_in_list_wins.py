# BUG: returns the first stop string of the list that occurs in the buffer instead of the match that completes first, so with two stops in one piece the cut can be too late.
def stop_step(state, piece: str, stops: list[str]) -> tuple[str, str, bool]:
    buffer = (state or "") + piece
    for stop in stops:
        start = buffer.find(stop)
        if start != -1:
            return "", buffer[:start], True
    hold = 0
    for stop in stops:
        for n in range(min(len(stop) - 1, len(buffer)), hold, -1):
            if buffer.endswith(stop[:n]):
                hold = n
                break
    cut = len(buffer) - hold
    return buffer[cut:], buffer[:cut], False


def stop_flush(state) -> str:
    return state or ""
