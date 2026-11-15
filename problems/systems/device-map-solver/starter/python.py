def solve_device_map(
    modules: list[tuple],
    max_memory: dict[int | str, int],
    no_split: set[str],
    tied: dict[str, str],
) -> dict[str, int | str]:
    """Assign the modules of a model, in order, to devices with limited memory.

    modules: the top-level modules in forward order. A module is a tuple
        (name, cls, body): body is an int (a leaf: its size in bytes) or a list of
        child modules.
    max_memory: {device: bytes}, the devices in the order to fill them. "disk" is not
        a key; it comes last and has no limit.
    no_split: class names whose modules must stay on one device.
    tied: {name of a tied top-level leaf: name of the leaf that owns its weights}.
    Returns {module name: device} covering every leaf exactly once.
    """
    raise NotImplementedError
