# BUG: among the matches present in the buffer it picks the one that starts first, not the one that completes first; when a short stop sits inside a longer match the result depends on how the text was split into pieces.
def stop_step(state, piece: str, stops: list[str]) -> tuple[str, str, bool]:
    buffer = (state or "") + piece
    starts = [buffer.find(stop) for stop in stops if stop in buffer]
    if starts:
        return "", buffer[: min(starts)], True
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
