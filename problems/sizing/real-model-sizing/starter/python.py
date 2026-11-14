def size_report(header: dict, config: dict) -> dict[str, int]:
    """Size a checkpoint from its safetensors header and its config.json.

    header: tensor name -> {"dtype": str, "shape": list[int], "data_offsets": [begin, end]},
        the tensor table of a safetensors file (what read_header returns first).
    config: the parsed config.json. Keys used: "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "hidden_size",
        "tie_word_embeddings", and, for a quantized checkpoint, "quantization"
        ({"bits": int, "group_size": int}).
    Returns a dict of Python ints with exactly these keys:
        "parameters": parameters of the model (packed weights unpacked, a tied head once)
        "tensor_bytes": bytes of the file's data section
        "kv_bytes_per_token": bytes one token adds to a bfloat16 KV cache
        "quantized_bytes": data bytes of an 8-bit, group-64 quantization of the model
    """
    raise NotImplementedError
