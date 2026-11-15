def allocate_bits(layers: list[dict], options: list[int], target_bpw: float) -> list[int]:
    """Choose one bit width per layer: lowest total error within a size budget.

    layers: one dict per layer, {"params": int, "errors": list[int]}; errors[k] is the
    layer's error when it is quantized to options[k] bits.
    options: distinct bit widths in increasing order. target_bpw: the largest allowed
    average number of bits per weight over all layers.
    Returns the chosen bit width of every layer, in layer order. Among assignments of
    equal total error: the fewest total bits, then the lexicographically smallest list.
    Raises ValueError if no assignment fits the budget.
    """
    raise NotImplementedError
