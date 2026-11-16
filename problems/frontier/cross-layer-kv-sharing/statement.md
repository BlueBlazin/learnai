Run a decoder in which only the lower layers keep a KV cache: every layer from
`share_from` up computes its own queries but attends over the keys and values that
layer `share_from - 1` has already stored. The upper layers then store nothing, and
that changes what a prompt costs: a layer that leaves nothing behind has no reason to
run on a position whose output nobody reads.

## Specification

```python
def prefill_shared(lm, prompt, share_from): ...  # -> (logits, cache)
def decode_step(lm, token, cache, share_from): ...  # -> (logits, cache)
```

`lm` is a [`LayeredLM`](../../../python/src/learnai/kit/layeredlm.py), a decoder
with $L$ = `lm.n_layers` layers whose stack can be run one range at a time:

```python
h = lm.embed(tokens, offset=n)  # (T, d): tokens at positions n, n + 1, ...
h, cache = lm.layers(h, a, b, cache)  # layers a .. b - 1, each storing its own K and V
h, cache = lm.layers(h, a, b, cache, kv_from=j)  # layers a .. b - 1 as readers of layer j
logits = lm.head(lm.norm(h))  # (T, vocab_size)
```

- `lm.layers(h, a, b, cache)` runs layers `a` to `b - 1` on hidden states `h` and
  returns the new states and a new cache; nothing is modified in place. Each of
  these layers appends the keys and values of the new positions to its own cache.
  Without a cache (`lm.layers(h, a, b)`) the layers start empty.
- With `kv_from=j` (a layer below `a`) the layers `a` to `b - 1` are **readers**:
  each computes queries from its input, attends over the keys and values that
  layer `j` has cached, and stores none of its own. The `T` rows of `h` are taken
  to be the **last** `T` positions layer `j` has cached, so layer `j` must have
  processed those positions already, in the cache that is passed.
- `cache.lengths` is the number of tokens each layer has cached and `cache.length`
  their maximum. Positions are learned embeddings: a token that continues a cached
  sequence must be embedded with `offset=` its position.
- `lm.counters` counts `layer_calls` (one per layer per call of `lm.layers`,
  whatever the number of tokens) and `layer_tokens` (layers times the tokens they
  ran on). `embed`, `norm`, and `head` are free.

**The model.** With $s$ = `share_from`, the sharing model is: layers $0, \dots, s-1$
are ordinary layers; layers $s, \dots, L-1$ are readers of layer $s-1$. Its logits at
a position are `lm.head(lm.norm(...))` of that position's state after layer $L-1$.
This is a different function from the ordinary model (`lm(tokens)`); a model is
trained for one or the other.

**`prefill_shared(lm, prompt, share_from)`** processes a prompt of $P \ge 1$ tokens
and returns

- `logits`: the sharing model's next-token logits after the prompt, shape
  `(vocab_size,)`, and
- `cache`: a cache in which layers $0, \dots, s-1$ hold $P$ tokens and layers
  $s, \dots, L-1$ hold none.

**`decode_step(lm, token, cache, share_from)`** feeds one more token, the one at
position `cache.length`, and returns the next-token logits after it and the cache
extended by that token (in layers $0, \dots, s-1$ only). The cache passed in is not
modified.

**Cost limits.**

| call | `layer_tokens` | `layer_calls` | why |
|---|---|---|---|
| `prefill_shared` | $s \cdot P + (L - s)$ | $L$ | the lower layers on every prompt token, in one call; the upper layers on the last position only |
| `decode_step` | $L$ | $L$ | every layer on the one new token |

The upper layers' output at a prompt position is needed for nothing: it is not a
logit anyone reads, and since those layers store no keys and values, no later
position attends to anything they produced. Only the last prompt position must go
through them.

**Stored KV.** After $n$ tokens the returned cache holds keys and values for $s$
layers: $2 \cdot s \cdot n \cdot d$ float32 numbers with $d$ = `lm.d_model`, which
is $8\,s\,n\,d$ bytes. The ordinary model stores $8\,L\,n\,d$.

## Shapes

| name | shape | dtype |
|---|---|---|
| `prompt` | list of $P$ ints | Python `int` |
| `token` | scalar | Python `int` |
| hidden states `h` | `(T, d_model)` for `T` positions | float32 |
| `logits` (returned) | `(vocab_size,)` | float32 |
| `cache` | a `LayerCache`, as returned by `lm.layers` | |

## Example

`LayeredLM(seed=3)` has $L = 4$ layers. With `share_from = 2`, layers 2 and 3 read
layer 1:

```python
logits, cache = prefill_shared(lm, [0, 17, 42, 5], 2)
int(logits.argmax())  # 48
cache.lengths  # (4, 4, 0, 0)
# layer_tokens: 2 * 4 + 2 = 10        layer_calls: 4

logits, cache = decode_step(lm, 48, cache, 2)
int(logits.argmax())  # 30
cache.lengths  # (5, 5, 0, 0)
# layer_tokens: 4                     layer_calls: 4
```

| | positions run by layers 0, 1 | positions run by layers 2, 3 | layer-tokens |
|---|---|---|---|
| `prefill_shared` | 0, 1, 2, 3 | 3 | $2 \cdot 4 + 2 \cdot 1 = 10$ |
| `decode_step` | 4 | 4 | $2 + 2 = 4$ |
| ordinary model, prefill | 0, 1, 2, 3 | 0, 1, 2, 3 | $4 \cdot 4 = 16$ |

In the prefill, layers 2 and 3 run on position 3 alone, and their attention still
covers the whole prompt: the keys and values of positions 0 to 3 are layer 1's.
For a prompt of 1,000 tokens and a model that shares from the middle, the same
table reads 2,002 layer-tokens against 4,000. (The ordinary model, `lm(tokens)`,
predicts 28 after this prompt, not 48: it is another function.)

## Constraints

- $2 \le L \le 6$, $1 \le s < L$, $1 \le P$, and the sequence never exceeds
  `lm.max_context` (128).
- Logits are compared with the kit's forward pass of the sharing model over the
  whole sequence, `lm(tokens, share_from=s)`, with `rtol = 1e-3` and `atol = 1e-4`
  (a cached and an uncached pass differ by about `1e-5`).
- After every call the tests check the logits, then `cache.lengths`, then the two
  counters of that call against the limits above. In the stored-KV cases the bytes
  of all key and value arrays in the final cache are compared with $8\,s\,n\,d$.
- The tests choose the tokens that are fed; your functions never pick one.

## Why this matters

KV memory is what limits how many long sequences fit on a GPU, and prefill is what
a user waits for before the first token. Sharing keys and values across layers
attacks both at once, which is why it has moved from papers into shipped models:
the upper half of the stack costs no cache at all, and a prompt only has to pass
through the lower half. But the saving is not automatic. An engine that runs the
full stack over the prompt out of habit produces the same logits and the same
cache, and gives the whole prefill speedup away; nothing but a profile shows it.
And an engine that lets the upper layers allocate a cache "like every layer" has
quietly turned the model into a different one.
