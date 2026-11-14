# BUG: counts one bfloat16 scale per group of 64 weights but not the bias stored next to it.
from math import prod

EMBEDDING = "model.embed_tokens.weight"
GROUP = 64  # weights per scale and bias in the predicted quantization


def size_report(header: dict, config: dict) -> dict[str, int]:
    quant = config.get("quantization")  # set when the linear layers are stored packed

    def is_packed(name: str) -> bool:
        """`X.weight` holding quantization codes: uint32 words with `X.scales` beside it."""
        base, _, leaf = name.rpartition(".")
        return (
            quant is not None
            and leaf == "weight"
            and header[name]["dtype"] == "U32"
            and f"{base}.scales" in header
        )

    def is_side_data(name: str) -> bool:
        """`X.scales` or `X.biases` of a packed `X.weight`: stored, but not parameters."""
        base, _, leaf = name.rpartition(".")
        weight = f"{base}.weight"
        return leaf in ("scales", "biases") and weight in header and is_packed(weight)

    # A tied model has one matrix for the embedding and the head. Some files store it
    # under both names anyway; the second copy is not a second set of parameters.
    duplicate_head = (
        bool(config["tie_word_embeddings"]) and EMBEDDING in header and "lm_head.weight" in header
    )

    parameters = 0
    quantized_bytes = 0
    for name, info in header.items():
        if is_side_data(name) or (duplicate_head and name.startswith("lm_head.")):
            continue
        shape = list(info["shape"])
        if is_packed(name):
            shape[-1] = shape[-1] * 32 // quant["bits"]  # codes per 32-bit word
        count = prod(shape)
        parameters += count
        if len(shape) == 2:
            # One byte per weight, plus a bfloat16 scale and a bfloat16 bias for every
            # group of 64 weights along a row: 8.5 bits per weight.
            quantized_bytes += count + 2 * shape[0] * (shape[1] // GROUP)
        else:
            quantized_bytes += 2 * count  # norm gains stay bfloat16

    # What is on disk is what the offsets say, whatever the dtype and the duplicates.
    tensor_bytes = sum(
        info["data_offsets"][1] - info["data_offsets"][0] for info in header.values()
    )

    head_dim = config["hidden_size"] // config["num_attention_heads"]
    # Keys and values, per layer, per KV head (not per query head), two bytes each.
    kv_bytes_per_token = (
        2 * config["num_hidden_layers"] * config["num_key_value_heads"] * head_dim * 2
    )

    return {
        "parameters": parameters,
        "tensor_bytes": tensor_bytes,
        "kv_bytes_per_token": kv_bytes_per_token,
        "quantized_bytes": quantized_bytes,
    }
