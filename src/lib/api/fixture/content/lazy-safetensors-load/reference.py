import json


def load_tensors(file, names: list[str]) -> dict[str, bytes]:
    """Return {name: bytes} reading only the header and the named tensors."""
    header_size = int.from_bytes(file.read(8), "little")
    header = json.loads(file.read(header_size))
    out = {}
    for name in names:
        begin, end = header[name]["data_offsets"]
        file.seek(8 + header_size + begin)
        out[name] = file.read(end - begin)
    return out
