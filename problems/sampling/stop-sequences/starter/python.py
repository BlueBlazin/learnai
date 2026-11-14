def stop_step(state, piece: str, stops: list[str]) -> tuple[object, str, bool]:
    """Push one piece of generated text through the stop-sequence filter.

    state: None on the first call of a stream, afterwards whatever the previous
        call returned as its first element. Its type is yours to choose.
    piece: the text of the newest token(s); may be the empty string.
    stops: the stop strings (non-empty strings; the list may be empty). The same
        list is passed on every call of a stream.
    Returns (state, emit, stopped): the new state, the text that may be shown to
    the user now, and whether a stop string has been completed. When stopped is
    True, emit ends exactly before the stop string and the stream is over.
    """
    raise NotImplementedError


def stop_flush(state) -> str:
    """The text still held back when the stream ends without a stop.

    state: the state returned by the last stop_step call (None if there was none).
    """
    raise NotImplementedError
