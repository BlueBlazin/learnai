## The idea

An inference runtime is a chain of representation changes: disk bytes to tensors, text to token ids, ids to logits, logits to a sampled token, and tokens back to text. Because almost every bug in this pipeline fails silently rather than crashing, the right strategy is to verify each stage's boundary output against the spec before moving to the next.

![Three files go in and streamed text comes out. Each box is labeled with its stage.](assets/architecture.svg)

**Stage 1** combines `safetensors-parse` and `mmap-lazy-loader`. Read the 8-byte header length, parse the JSON header, validate every requested name, and then slice only the byte ranges of the requested tensors. Wrapping each slice with `np.frombuffer` avoids reading untouched pages, so the only copy is widening 16-bit floats to `float32`. Because `BF16` is the upper 16 bits of IEEE `float32`, shifting each `uint16` left by 16 bits and viewing as `float32` is exact. Both common loader bugs fail silently. Counting `data_offsets` from byte 0 reads header JSON into tensors, and interpreting `BF16` as `F16` produces plausible small numbers because both formats share the sign bit position.

**Stage 2** combines `tokenizer-json-load` with Granite's chat template from `real-model-chat-turn`. Operation order in `encode` is critical. First, split out added tokens from raw text so markers like `<|start_of_role|>` become single ids. Next, apply the pre-tokenizer regex so BPE merges never cross word boundaries. Finally, map UTF-8 bytes to characters and run rank-ordered BPE per word. Store both an id-to-bytes lookup table and the set of special token ids on the `Tokenizer` instance for stage 5.

**Stage 3** runs Granite's forward pass driven entirely by `config.json`. Granite differs from a textbook Llama block in several config-controlled ways: `embedding_multiplier`, `residual_multiplier` on both branches, `attention_multiplier` (`1 / head_dim` on the real model rather than `1 / sqrt(head_dim)`), `logits_scaling`, a fused SwiGLU input matrix with gate rows first, and tied embeddings. Because the tiny model changes every config value, hardcoding any constant fails on one of the two models.

**Stage 4** adds a KV cache and deterministic sampling. Share one helper function that runs a sequence of tokens after an optional cache. Positions start at `cache["length"]`, new keys are rotated for their absolute positions and stored rotated at key/value-head width, and the causal mask lets position `p` attend to slots `0..p`. `forward` calls this helper with `cache=None` and projects all rows to logits, while `prefill` and `decode_step` project only the last row. `sample_token` applies penalties on raw logits, runs top-k, top-p, and min-p sequentially, scales by temperature last, and draws with an inverse CDF over the caller's uniform `u`.

**Stage 5** chains two streaming state machines. Python's incremental UTF-8 decoder holds back trailing incomplete multi-byte characters, while the stop-string filter holds back the longest suffix that is a proper prefix of any stop string. Keeping the full decoded string `text` alongside an index `shown` reduces the visibility rule to one helper: cut at the first match to complete, or hold back the matching prefix suffix, or release everything when generation finishes.

## Reference solution (PyTorch)

```python
def load_weights(file, names=None):
    n = int.from_bytes(file[0:8], "little")
    header = json.loads(bytes(file[8 : 8 + n]))
    header.pop("__metadata__", None)
    data_start = 8 + n  # every data_offsets pair counts from here
    names = list(header) if names is None else list(names)
    for name in names:
        if name not in header:
            raise KeyError(name)  # before any tensor byte is read
    weights = {}
    for name in names:
        entry = header[name]
        begin, end = entry["data_offsets"]
        stored = np.frombuffer(
            file[data_start + begin : data_start + end], FLOAT_DTYPES[entry["dtype"]]
        )
        if entry["dtype"] == "BF16":
            values = (stored.astype(np.uint32) << 16).view(np.float32)
        else:
            values = stored.astype(np.float32)
        weights[name] = torch.from_numpy(values.reshape(entry["shape"]))
    return weights


def _layers(weights, cfg, tokens, cache):
    seq = tokens.shape[0]
    past = 0 if cache is None else cache["length"]
    half, group = cfg.head_dim // 2, cfg.n_heads // cfg.n_kv_heads
    positions = past + torch.arange(seq)  # continue where the cache ends
    inv_freq = cfg.rope_theta ** (-torch.arange(half, dtype=torch.float32) / half)
    angles = positions.to(torch.float32)[:, None] * inv_freq
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
    visible = torch.arange(past + seq)[None, :] <= positions[:, None]

    def rope(z):
        return z * cos + torch.cat([-z[..., half:], z[..., :half]], dim=-1) * sin

    def heads(z, n):
        return z.reshape(seq, n, cfg.head_dim).transpose(0, 1)

    x = weights["model.embed_tokens.weight"][tokens] * cfg.embedding_multiplier
    keys, values = [], []
    for i in range(cfg.n_layers):
        p = f"model.layers.{i}."
        h = ops.rms_norm(x, weights[p + "input_layernorm.weight"], cfg.norm_eps)
        q = rope(heads(h @ weights[p + "self_attn.q_proj.weight"].T, cfg.n_heads))
        k = rope(heads(h @ weights[p + "self_attn.k_proj.weight"].T, cfg.n_kv_heads))
        v = heads(h @ weights[p + "self_attn.v_proj.weight"].T, cfg.n_kv_heads)
        if cache is not None:  # cat builds new tensors: the old cache is untouched
            k = torch.cat([cache["keys"][i][:, :past], k], dim=1)
            v = torch.cat([cache["values"][i][:, :past], v], dim=1)
        keys.append(k)
        values.append(v)
        scores = q @ k.repeat_interleave(group, dim=0).transpose(-1, -2)
        scores = (scores * cfg.attention_scale).masked_fill(~visible, float("-inf"))
        mixed = ops.softmax(scores) @ v.repeat_interleave(group, dim=0)
        mixed = mixed.transpose(0, 1).reshape(seq, cfg.d_model)
        x = x + cfg.residual_multiplier * (mixed @ weights[p + "self_attn.o_proj.weight"].T)
        h = ops.rms_norm(x, weights[p + "post_attention_layernorm.weight"], cfg.norm_eps)
        both = h @ weights[p + "shared_mlp.input_linear.weight"].T  # gate, then up
        hidden = torch.nn.functional.silu(both[..., : cfg.d_ff]) * both[..., cfg.d_ff :]
        x = x + cfg.residual_multiplier * (
            hidden @ weights[p + "shared_mlp.output_linear.weight"].T
        )
    return x, {"length": past + seq, "keys": keys, "values": values}


def sample_token(logits, sampler, history, u):
    occurrences = np.bincount(np.asarray(history, dtype=np.int64), minlength=logits.shape[-1])
    counts = torch.from_numpy(occurrences.astype(np.float32))
    seen = counts > 0
    penalty = sampler["repetition_penalty"]
    x = torch.where(seen, torch.where(logits > 0, logits / penalty, logits * penalty), logits)
    x = x - counts * sampler["frequency_penalty"] - seen * sampler["presence_penalty"]
    x = ops.top_k_filter(x, sampler["top_k"])
    if sampler["top_p"] < 1.0:
        x = ops.top_p_filter(x, sampler["top_p"])
    x = ops.min_p_filter(x, sampler["min_p"])
    cdf = torch.cumsum(ops.softmax(x / sampler["temperature"]), dim=-1)
    return int((cdf <= u * cdf[-1]).sum())


def stream_chat(weights, config, tokenizer, messages, max_new_tokens, sampler, uniforms, stop):
    prompt_ids = torch.tensor(tokenizer.encode(chat_prompt(messages)))
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    text, shown = "", 0
    for token in _tokens(weights, config, prompt_ids, max_new_tokens, sampler, uniforms):
        text += decoder.decode(b"" if token in tokenizer.special else tokenizer.token_bytes(token))
        end, stopped = _visible(text, stop, final=False)
        if end > shown:
            yield text[shown:end]
            shown = end
        if stopped:
            return  # closes the token generator: no further forward pass
    text += decoder.decode(b"", final=True)
    end, _ = _visible(text, stop, final=True)
    if end > shown:
        yield text[shown:end]
```

`_tokens` is the generator shown in the third hint, and `_visible(text, stops, final)` returns `(end, stopped)`. The tokenizer and chat template follow the two prerequisite problems. The JAX reference differs in one place: it pads prompts and caches to multiples of 32 slots and masks unused positions so JAX compiles only a few fixed shapes.

## Common mistakes

- **Safetensors loader bugs.** Counting offsets from byte 0 (`offsets_from_file_start`), reading `BF16` as IEEE `F16` (`bf16_as_float16`), or reading the whole file when only a few tensors are requested (`reads_whole_file`).
- **Skipping pre-tokenization** (`ignores_pretokenizer`). Merges cross word boundaries, causing ids to diverge around repeated spaces or digit runs longer than three.
- **Wrong chat template.** Using ChatML (`chatml_template`) or omitting the default system turn when the conversation has no system message (`default_system_missing`).
- **Hardcoding textbook transformer math.** Using `1 / sqrt(head_dim)` (`attention_scale_sqrt`), pairing interleaved RoPE dimensions instead of half-split (`rope_interleaved`), skipping `residual_multiplier` on the MLP branch (`mlp_residual_unscaled`), or splitting the fused MLP matrix with `up` before `gate` (`up_rows_first`).
- **Restarting decode positions at zero** (`new_token_position_zero`). This may still produce readable text with your own cache, but fails immediately against a test-provided cache.
- **Wrong cache layout or aliasing.** Expanding cached keys and values to query-head width (`cache_at_query_width`, 4x memory on the real model), or mutating a shared buffer across calls (`cache_shared_buffer`), which corrupts branched continuations.
- **Sampling and loop bugs.** Applying temperature before the filters (`temperature_first`), including `prompt_ids` in `history` (`history_includes_prompt`), or returning `eos_token_id` in the output list (`eos_returned`).
- **Streaming bugs.** Decoding each token independently instead of incrementally (`decodes_tokens_alone`), emitting bytes for special tokens (`special_tokens_shown`), checking for stop strings only inside the newest piece (`stop_within_piece`), yielding text that could start a stop string (`stop_prefix_shown`), or dropping held-back text when generation finishes (`held_text_dropped`).
- **Using the kit's tokenizer** (`uses_kit_tokenizer`). Every stage must be your own implementation.

## Complexity

Loading reads each requested tensor's bytes once: 705 MB on disk for the real model, expanding to 1.4 GB in `float32`. Prefill over $P$ tokens runs $P$ tokens through all layers with $O(P^2)$ attention scores per head. Each decode step runs one token through all layers with $O(n)$ attention scores against a cache of length $n$. The cache stores $2 \cdot L \cdot H_{kv} \cdot \texttt{head\_dim}$ floats per token (14,336 floats, or 57 kB in `float32` on the real model). Sampling costs a sort over the vocabulary unless `top_k` prunes it first. Rescanning the full decoded text for stop strings is quadratic in reply length; a production runtime only scans the last `max(len(s) for s in stop) - 1` characters plus the newly decoded text.

## In the wild

The safetensors binary layout is documented at https://huggingface.co/docs/safetensors. The tokenizer pipeline (added tokens, pre-tokenizer, byte-level BPE) matches Hugging Face `tokenizers`, and Granite's chat format comes from the Jinja template in https://huggingface.co/ibm-granite/granite-4.0-350m. `mlx-lm` structures generation the same way, with a prompt prefill followed by single-token decode steps over a per-layer KV cache (https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py). The sampler stage order matches the default chain in `llama.cpp` (https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
