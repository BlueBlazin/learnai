Each stage is an earlier problem with its interface changed a little, and each of
them fails in a way that still produces output. So before writing stage N, ask what
stage N - 1 hands it and how you would notice if that were wrong: which offset do
`data_offsets` count from? Which of the model's numbers come from `config`, and
which from a weight's shape? What does a decode step know about the tokens before
it, if all it gets is one token and the cache? And when a stop string arrives in two
tokens, what must already have happened to its first half?

---

Stage 1: `n = int.from_bytes(file[0:8], "little")`, parse `bytes(file[8:8 + n])`,
check the names, then one slice `file[8 + n + begin : 8 + n + end]` per requested
tensor, wrapped with `np.frombuffer`; bfloat16 is `(bits.astype(np.uint32) <<
16).view(np.float32)`. Stage 2 is `tokenizer-json-load` as a class that also keeps a
dict from id to bytes and the set of special ids, plus the three-part template of
`real-model-chat-turn`. Stage 3 and stage 4 share one function, "run these tokens
after this cache": positions start at `cache["length"]`, new keys are rotated for
those positions and appended to the cached ones at key/value width, and the mask
lets position `p` see slots `0..p`. `forward` calls it without a cache and puts
every row through the head; `prefill` and `decode_step` put the last row through.
The sampler is `sampler-chain` for one row, with `history` as a count per token.

---

Stage 4's loop, written as a generator so that stage 5 can stop it:

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

`generate` is `list(tokens(...))`. For stage 5, keep the whole decoded text and a
count `shown` of characters already yielded. After each token, append
`decoder.decode(token_bytes)` (empty bytes for a special token) and compute where
the visible text ends: at the start of the match with the smallest `(end, start)`
if a stop string occurs in the text (then yield and return), else at `len(text)`
minus the longest suffix that equals `stop[:k]` for some stop and some `k <
len(stop)`. Yield `text[shown:end]` when it is not empty. After the loop, add
`decoder.decode(b"", final=True)` and do the same once more without holding
anything back.
