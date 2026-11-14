def complete_prefix_len(data: bytes) -> int:
    n = len(data)
    # A pending sequence is a lead byte plus at most two continuation bytes, so only
    # the last three bytes can belong to one. Walk back to the first byte that is not
    # a continuation byte: it alone decides.
    for back in range(1, min(3, n) + 1):
        byte = data[n - back]
        if byte & 0xC0 == 0x80:
            continue  # 10xxxxxx: a continuation byte; its lead byte is further left
        if back < _sequence_length(byte) and _second_byte_ok(data, n - back, back):
            return n - back  # an unfinished character that can still become valid
        return n  # ASCII, a finished character, or bytes that can never be valid
    return n  # empty, or three continuation bytes: nothing here is waiting for more


def _sequence_length(lead: int) -> int:
    """Bytes in the character that `lead` starts; 1 for ASCII and for invalid leads."""
    if 0xC2 <= lead <= 0xDF:
        return 2
    if 0xE0 <= lead <= 0xEF:
        return 3
    if 0xF0 <= lead <= 0xF4:
        return 4
    return 1  # ASCII, a continuation byte, or C0, C1, F5..FF, which start nothing


def _second_byte_ok(data: bytes, start: int, have: int) -> bool:
    """Whether the byte after the lead (if it has arrived) is in the lead's range."""
    if have < 2:
        return True
    lead, second = data[start], data[start + 1]
    if lead == 0xE0:
        return second >= 0xA0  # below: an overlong encoding
    if lead == 0xED:
        return second <= 0x9F  # above: a UTF-16 surrogate
    if lead == 0xF0:
        return second >= 0x90  # below: an overlong encoding
    if lead == 0xF4:
        return second <= 0x8F  # above: past U+10FFFF
    return True
