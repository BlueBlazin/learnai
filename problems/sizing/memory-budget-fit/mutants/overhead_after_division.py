# BUG: the overhead is applied after the division: the blocks are counted on the whole device minus the weights and then scaled by (100 - percent) / 100, so the weights are treated as if the reserve shrank them too.
ITEMSIZE = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1}


def pool_blocks(
    memory_bytes: int, weight_bytes: int, cfg: dict, dtype: str, overhead_percent: int
) -> int:
    """KV blocks that fit once the reserve and the weights are taken out of the device."""
    # The reserve is a share of the whole device, rounded up: ceil(memory * percent / 100).
    cache_bytes = memory_bytes - weight_bytes
    token_bytes = 2 * cfg["n_layers"] * cfg["n_kv_heads"] * cfg["head_dim"] * ITEMSIZE[dtype]
    block_bytes = cfg["block_tokens"] * token_bytes
    # A partial block is no block; a model that does not fit leaves no pool at all.
    return max(0, cache_bytes // block_bytes * (100 - overhead_percent) // 100)


def max_sequences(
    memory_bytes: int, weight_bytes: int, cfg: dict, dtype: str, context: int, overhead_percent: int
) -> int:
    blocks = pool_blocks(memory_bytes, weight_bytes, cfg, dtype, overhead_percent)
    # A sequence holds whole blocks: its last, partly filled block counts in full.
    blocks_per_sequence = -(-context // cfg["block_tokens"])
    return blocks // blocks_per_sequence


def max_context(
    memory_bytes: int, weight_bytes: int, cfg: dict, dtype: str, batch: int, overhead_percent: int
) -> int:
    blocks = pool_blocks(memory_bytes, weight_bytes, cfg, dtype, overhead_percent)
    # Every sequence gets the same number of whole blocks; leftover blocks help nobody.
    return (blocks // batch) * cfg["block_tokens"]
