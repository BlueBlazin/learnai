## The idea

In a causal transformer, the keys and values of position $i$ depend on tokens $0..i$
and on nothing after them. Two prompts that agree on their first $n$ tokens therefore
have identical keys and values at positions $0..n-1$. A cache computed for one prompt
is a valid cache for the other, as far as their common prefix reaches, and invalid
from the first differing token on.

That gives the three steps:

1. **Match.** Compare the prompt with every stored key and take the longest common
   prefix, $m$ tokens.
2. **Truncate.** The stored cache may cover more than those $m$ tokens (the old
   prompt went on, possibly differently). `truncate` cuts it back to the shared part.
   Without this step the leftover positions stay in the context, and the model
   answers "shared part + old tail + new tail".
3. **Feed the rest.** The remaining $P - m$ prompt tokens go in as one call on top of
   the truncated cache, and its last row of logits starts the decode loop.

One corner needs care. If the prompt is itself a stored key, or a prefix of one, then
$m = P$ and nothing remains to feed. But a cache holds keys and values, not logits:
the only way to learn the first generated token is to run the last prompt token
through the model. So reuse $r = \min(m, P - 1)$ tokens and feed the last prompt
token again. The same expression covers the single-token prompt ($r = 0$).

The entry to store is the cache right after the prefill call, which covers exactly
the prompt. Since `TinyLM` returns a new cache from every call, the decode loop does
not disturb it.

## Reference solution (PyTorch)

```python
def common_prefix_len(a, b):
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    return n


def generate_with_store(lm, store, prompt, max_new_tokens):
    best_key, best = None, 0
    for key in store:
        shared = common_prefix_len(key, prompt)
        if shared > best:
            best_key, best = key, shared
    reused = min(best, len(prompt) - 1)

    if reused > 0:
        cache = store[best_key].truncate(reused)
        logits, cache = lm(prompt[reused:], cache)
    else:
        logits, cache = lm(prompt)
    new_store = {**store, tuple(prompt): cache}

    generated = []
    while True:
        token = int(torch.argmax(logits[-1]))
        generated.append(token)
        if len(generated) == max_new_tokens:
            return generated, new_store
        logits, cache = lm([token], cache)
```

Why the limits are tight: the model must see every prompt token the store does not
cover, the store covers at most $m$ of them, and at least one call on a prompt token
is needed for logits. Hence at least $P - \min(m, P - 1)$ prompt tokens, plus the
$g - 1$ generated tokens that later tokens depend on.

## Common mistakes

- **Reusing only on an exact match** (`if tuple(prompt) in store`). It is the obvious
  dictionary lookup and it misses almost every real hit: a follow-up turn extends the
  stored prompt, and a different question after the same system prompt shares a
  prefix with it, but neither is equal to a key.
- **Reusing only whole entries** (the key must be a prefix of the prompt). This
  handles the growing conversation and misses the fork: two questions after one
  system prompt share that system prompt, but each stored key continues past it.
- **Not truncating.** Continuing from the stored cache as it is works whenever the
  entry is entirely a prefix of the prompt, so it survives simple tests. When the
  entry diverges, the cost looks perfect and the tokens are wrong.
- **Feeding nothing on a full match.** With $r = m = P$ the call gets an empty token
  list, which the model rejects. The same bug hides in `truncate(P)` followed by a
  decode loop that expects logits to exist.
- **Always holding one token back** ($r = m - 1$). It repairs the previous bug by
  paying one extra token on every request instead of only on full matches.
- **Taking the first match instead of the longest.** Correct tokens, wasted work, and
  the waste depends on dictionary order, so it shows up only under load.
- **Storing the wrong cache.** Saving the cache after generation under the key
  `tuple(prompt)` breaks the invariant that an entry covers exactly its key: the next
  request that extends the prompt would continue after the old reply. Dropping the
  old entries from the returned store throws away other users' prefixes.

## Complexity

Matching costs $O(\sum_{\text{keys}} \min(|key|, P))$ token comparisons, the model
processes $P - r + g - 1$ tokens in $g$ calls, and the store grows by one cache of
$P$ tokens. The linear scan over keys is the part that does not scale; engines index
the store instead (see below).

## In the wild

- llama.cpp's server keeps the tokens of the last prompt per slot; with
  `cache_prompt` it compares the new prompt with them, keeps the common prefix in the
  KV cache, removes the rest, and evaluates only the new suffix.
- `mlx-lm`'s server keeps an LRU store of prompt caches and looks up the nearest
  entry. When a stored entry is longer than the match it trims a copy of it, with the
  prefix capped at `len(tokens) - 1`, the same "keep one token to feed" rule as here.
  It also stores the cache *with* the generated reply appended to the key, so the next
  turn of a conversation reuses the reply as well. This problem stores the prompt
  only, which is why the conversation in the tests reuses the previous prompt and
  re-processes the previous reply.
- vLLM's automatic prefix caching does not compare prompts at all: the KV cache lives
  in fixed-size blocks, each full block is identified by a hash of its tokens and of
  everything before it, and a new request reuses every leading block whose hash is
  already present. `block-hash-prefix-cache` builds that index, and
  `radix-prefix-cache` the tree-shaped alternative that SGLang uses.
- Truncation is cheap only because of causality. A cache of a model with a sliding
  window, a recurrent state, or evicted entries cannot always be rolled back to an
  arbitrary length, and engines then fall back to checkpoints or to recomputing.
