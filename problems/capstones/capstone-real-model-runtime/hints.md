Each stage adapts an earlier problem and can fail silently while still producing output. Before writing stage $N$, verify what stage $N - 1$ hands it. Check where `data_offsets` start in a safetensors file, which model hyperparameters come from `config` versus weight shapes, how `decode_step` determines the current token's position from the cache alone, and what must be held back when a stop string can span multiple tokens.

---

- **Stage 1:** Read `n = int.from_bytes(file[0:8], "little")` and parse `bytes(file[8 : 8 + n])`. Validate all requested names before slicing `file[8 + n + begin : 8 + n + end]` into `np.frombuffer`. Convert `BF16` with `(bits.astype(np.uint32) << 16).view(np.float32)`.
- **Stage 2:** Wrap `tokenizer-json-load` in a class that also stores an id-to-bytes mapping and the set of special token ids. Implement `chat_prompt` using the three-part Granite template from `real-model-chat-turn`.
- **Stages 3 and 4:** Share one internal helper that runs a token sequence after an optional cache. Positions start at `cache["length"]`, new keys are rotated for those positions and stored at key/value-head width, and the causal mask lets position `p` attend to slots `0..p`. `forward` runs without a cache and projects every row to logits, while `prefill` and `decode_step` project only the last row. `sample_token` is `sampler-chain` on a single logit vector, using token counts from `history`.

---

Write stage 4's generation loop as a generator so stage 5 can stop it early without extra forward passes:

```python
def tokens(weights, config, prompt_ids, max_new_tokens, sampler, uniforms):
    logits, cache = prefill(weights, config, prompt_ids)
    history = []
    for step in range(max_new_tokens):
        token = sample_token(logits, sampler, history, uniforms[step])
        if token == config["eos_token_id"]:
            return
        history.append(token)
        yield token
        if step + 1 < max_new_tokens:
            logits, cache = decode_step(weights, config, token, cache)
```

Then `generate` is `list(tokens(...))`. For stage 5, maintain the full decoded `text` and an index `shown` of characters already yielded. After each token, append `decoder.decode(b"" if special else token_bytes)` and compute the visible end index `end`. If any stop string occurs in `text`, pick the match with the smallest `(end_idx, start_idx)`, yield `text[shown:start_idx]` if non-empty, and return. Otherwise, set `end` to `len(text)` minus the longest suffix equal to `s[:k]` for some stop string `s` and `k < len(s)`, and yield `text[shown:end]` if non-empty. After the loop, append `decoder.decode(b"", final=True)` and run the same check once more with no prefix holdback.
