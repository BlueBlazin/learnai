## The idea

A runtime is a pipeline of representations, and each stage of this capstone is one
change of representation: bytes on disk to tensors, text to ids, ids to logits,
logits to a token, tokens to text. None of the five is hard. What makes the whole
thing a capstone is that every stage is correct only *relative to a reference*:
nothing crashes when it is wrong. So the method is the same at each step: fix the
representation at the boundary, and compare it with something independent before
building on it.

![The runtime. Stage 1 turns the checkpoint into float32 tensors, stage 2 turns a conversation into token ids, stage 3 is the forward pass, stage 4 wraps it in a cache and a sampler, and stage 5 turns the sampled tokens into text that is held back while it may still become a stop string.](assets/architecture.svg)

**Stage 1** is `safetensors-parse` and `mmap-lazy-loader` in one function. Read 8
bytes, read the header, and from then on touch only the byte ranges of the tensors
that were asked for; `np.frombuffer` over a slice of the map is a view, so the only
copy is the one that turns 16-bit patterns into float32. That conversion is exact:
bfloat16 *is* the upper half of a float32, so shifting the pattern left by 16 bits
gives the value, and nothing is rounded. The two classic errors are both silent.
Offsets counted from byte 0 instead of from the end of the header return tensors of
the right shape filled with JSON text and the tail of the previous tensor. And
bfloat16 read as IEEE float16 returns plausible small numbers, because the two
formats agree on the sign bit and differ in where the exponent ends.

**Stage 2** is `tokenizer-json-load` plus the template of `real-model-chat-turn`.
The order of operations in `encode` is the content: added tokens are cut out of the
raw text first (which is why `<|start_of_role|>` written in a prompt is one id), the
pre-tokenizer then decides where merges may not cross, and only then does BPE run,
per word, by rank. The class keeps what stage 5 needs: a table from id to bytes and
the set of special ids.

**Stage 3** is `real-model-logits` reading `config.json` itself. Everything unusual
about Granite is a number in that file or a shape in the checkpoint: the embedding
multiplier, the residual multiplier on *both* branches, the attention multiplier
(`1 / head_dim` in the real model, not `1 / sqrt(head_dim)`), the logits divisor,
the fused SwiGLU input matrix with the gate rows first, the tied head. The tiny
model has other values in every field, so a constant copied from the real config
fails there and a constant copied from the tiny one fails on the real model.

**Stage 4** gives the forward pass a memory. Write one function that runs "these
tokens after this cache": positions continue at `cache["length"]`, each new key is
rotated for its own position and stored rotated, keys and values are kept at the
width of the key/value heads and expanded to the query heads only for the matrix
product, and the mask lets position `p` see slots `0..p`. `forward` is that function
with no cache and the head on every row; `prefill` and `decode_step` take the last
row. Because `decode_step` receives one token and a cache, it cannot recompute
anything: if the cache is wrong, the logits are wrong. The sampler is the chain of
`sampler-chain`: penalties on raw logits, three truncation filters that each see
what the previous one left, temperature last, one draw by inverse CDF with a
supplied uniform number. With randomness as an input, sampled generation is as
reproducible as greedy decoding.

**Stage 5** is two small state machines in a row. The incremental UTF-8 decoder
holds back the bytes of an unfinished character; the stop filter holds back the
longest suffix of the text that could still grow into a stop string. Both exist for
the same reason: what has been yielded cannot be taken back. Keeping the whole
decoded text and an index `shown` makes the rules one function: where does the
visible text end? At the start of the first match to complete, or at the end minus
the held suffix, or, when the reply is over, at the very end.

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

`_tokens` is the generator in the last hint; `_visible(text, stops, final)` returns
the end of the visible text and whether a stop string completed. The tokenizer and
the template are those of the two prerequisites. The JAX reference differs in one
place: it pads prompts and caches to multiples of 32 slots and masks the unused
ones, so that JAX compiles a handful of shapes instead of one per length.

## Common mistakes

- **Offsets from the start of the file** (`offsets_from_file_start`), **bfloat16
  read as float16** (`bf16_as_float16`), **reading the whole file to return three
  tensors** (`reads_whole_file`).
- **Skipping the pre-tokenizer** (`ignores_pretokenizer`): merges then cross word
  boundaries, and ids differ wherever two spaces or more than three digits meet.
- **The wrong template**: ChatML (`chatml_template`), or no system turn when the
  conversation has none (`default_system_missing`).
- **Textbook constants instead of the config's**: `1 / sqrt(head_dim)`
  (`attention_scale_sqrt`), interleaved RoPE pairs (`rope_interleaved`), the
  residual multiplier on one branch only (`mlp_residual_unscaled`), the fused MLP
  matrix split the wrong way round (`up_rows_first`).
- **Positions that restart at zero in a decode step** (`new_token_position_zero`).
  With a cache you built yourself this still produces text; with a cache built
  elsewhere it is caught at the first step.
- **A cache at query-head width** (`cache_at_query_width`): four times the memory in
  the real model. **A cache that is a shared, mutable buffer**
  (`cache_shared_buffer`): fast, and wrong the first time two continuations start
  from one prefix.
- **Temperature before the filters** (`temperature_first`), **penalties that count
  the prompt** (`history_includes_prompt`), **returning the end-of-sequence token**
  (`eos_returned`).
- **Decoding tokens one at a time** (`decodes_tokens_alone`), **showing special
  tokens** (`special_tokens_shown`), **looking for a stop string only inside the
  newest token** (`stop_within_piece`), **showing text that may be the beginning of
  a stop string** (`stop_prefix_shown`), **forgetting to release held text at the
  end** (`held_text_dropped`).
- **Borrowing the kit's tokenizer** (`uses_kit_tokenizer`): the point is that
  nothing in the file is somebody else's.

## Complexity

Loading reads each requested tensor's bytes once: 705 MB for the real model, and
the float32 copy is twice that, 1.4 GB. A prefill of $P$ tokens costs $P$ tokens
through the layers and $O(P^2)$ attention scores per head; a decode step costs one
token through the layers and $O(n)$ scores, independent of how the prompt was read.
The cache holds $2 \cdot L \cdot H_{kv} \cdot \texttt{head\_dim}$ floats per token:
14,336 for the real model, 57 kB per token in float32. The sampler is a sort of the
vocabulary per token unless top-k comes first. The stop filter, as written, rescans
the text per token, which is quadratic in the reply; a production filter keeps only
the last `max(len(stop)) - 1` characters.

## In the wild

The file format is documented at https://huggingface.co/docs/safetensors (the
layout that stage 1 reads). The tokenizer pipeline
(added tokens, pre-tokenizer, byte-level BPE) is that of the `tokenizers` library,
and the chat layout is the Jinja template shipped with
https://huggingface.co/ibm-granite/granite-4.0-350m ; the fixtures of this problem
were checked against both and against `transformers`' implementation of the model.
`mlx-lm` structures generation the same way, as a prompt pass followed by
single-token steps over a per-layer key/value cache
(https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py), and the order
of the sampler stages is the default chain of llama.cpp
(https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
