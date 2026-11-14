import json


def load_tensors(file, names: list[str]) -> dict[str, bytes]:
    """Return {name: bytes} reading only the header and the named tensors."""
    raise NotImplementedError
