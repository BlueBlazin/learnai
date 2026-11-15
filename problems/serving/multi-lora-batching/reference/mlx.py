import mlx.core as mx


def batched_lora_forward(
    x: mx.array,
    base,
    a_stack: mx.array,
    b_stack: mx.array,
    ranks: mx.array,
    adapter_ids: mx.array,
    alpha: float,
) -> mx.array:
    y = base(x)  # the shared weight, once, for every row whatever its adapter

    # Row b uses adapter adapter_ids[b]. An id of -1 must not be used as an index
    # (it would select the last adapter): gather adapter 0 and scale its update by 0.
    has_adapter = adapter_ids >= 0
    index = mx.maximum(adapter_ids, 0)
    a = a_stack[index]  # (B, r_max, in): gathered on the adapter axis
    b = b_stack[index]  # (B, out, r_max)
    # Each adapter is scaled by its own rank; the padded rank is not a rank.
    scale = mx.where(has_adapter, alpha / ranks[index], 0.0).astype(x.dtype)  # (B,)

    tokens = x if x.ndim == 3 else x[:, None, :]  # (B, T, in)
    low = tokens @ mx.swapaxes(a, 1, 2)  # (B, T, r_max): down-projection per row
    delta = (low @ mx.swapaxes(b, 1, 2)) * scale[:, None, None]  # (B, T, out)
    return y + delta.reshape(y.shape)
