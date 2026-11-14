def complete_prefix_len(data: bytes) -> int:
    """How many leading bytes of `data` can be decoded now.

    data: any bytes, valid UTF-8 or not, possibly cut off in the middle of a character.
    Returns len(data) minus the length of the pending sequence at the end of `data`:
    a lead byte followed by fewer continuation bytes than it calls for, all of them
    allowed so far by the UTF-8 table in the statement. Returns len(data) if there is
    no pending sequence. Invalid bytes are never pending.
    """
    raise NotImplementedError
