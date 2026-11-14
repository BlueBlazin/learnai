Generate greedily for a batch of prompts of **different lengths** with one model call
per step. A batch is a rectangle, so the shorter prompts are padded, and the entire
problem is making the padding invisible: every row must produce exactly the tokens it
produces alone, while rows stop at their own EOS and the loop ends the moment the last
row is done.

## Specification

```python
def batch_generate(model, prompts, max_new_tokens, eos_id): ...
```

`model` is an [`MLXTinyLM`](../../../python/src/whetstone/kit/mlxlm.py), the kit's
small seeded transformer in `mlx.nn`. On a batch it takes two extra arguments:

```python
logits, cache = model(tokens, mask=mask, positions=positions)  # the whole rows so far
logits, cache = model(tokens, cache, mask=mask, positions=positions)  # continue the rows
```

- `tokens`: integer array of shape `(B, T)`, $T \ge 1$.
- `mask`: a float array that is **added** to the attention scores before the softmax,
  in place of the model's causal mask. It must broadcast to `(B, 1, T, C + T)`, where
  $C$ is the number of columns already in the cache (0 without one): entry
  `[b, 0, i, j]` is `0` where query $i$ of this call may attend to key column $j$ of
  row $b$, and `-inf` where it may not. Key columns are numbered across calls: the
  cached columns first, then the columns of this call.
- `positions`: integer array of shape `(B, T)`: the position of each token of this
  call, used for RoPE (or for the sinusoidal embedding of a model built with
  `rope=False`).
- `logits`: float32, shape `(B, T, vocab_size)`; `logits[b, -1]` scores the next
  token of row `b`. `cache` holds all `B` rows; a call with a cache must pass the same
  number of rows. Both are **lazy**, and the model never modifies a cache.

**What to compute.** Write $\operatorname{solo}(p)$ for greedy decoding of the single
prompt $p$: positions $0, 1, 2, \dots$, a causal mask, and the argmax of the
next-token logits at every step. Row `b` of the result is the tokens
$\operatorname{solo}(\texttt{prompts}[b])$ generates until the row is *finished*:

- it has generated `eos_id` (**the EOS token is the last element of the row**), or
- it has generated `max_new_tokens` tokens.

Arguments and result:

- `prompts`: a list of $B \ge 1$ lists of token ids, of lengths $L_b \ge 1$. They
  are not modified.
- `max_new_tokens`: an int $\ge 1$, the limit for every row.
- `eos_id`: an int or `None`. Only generated tokens count: an `eos_id` inside a
  prompt stops nothing. With `None` every row gets `max_new_tokens` tokens.
- Returns a list of $B$ lists of Python ints: the generated tokens of each row,
  without the prompt. Row $b$ has $1 \le g_b \le$ `max_new_tokens` tokens.

**How to call the model.** Let $T = \max_b L_b$, $P_b = T - L_b$, and
$g_{\max} = \max_b g_b$.

1. **One call for the prompts**, padded **on the left** to shape $(B, T)$: row $b$ is
   $P_b$ padding slots followed by `prompts[b]` (Figure 1). Whether a column is
   padding is decided by the lengths, never by a token's value: any token id may fill
   the slots, and the same id may occur in a prompt as a real token.
2. **Mask.** A real query (column $i \ge P_b$) may attend to key column $j$ exactly
   when $j \le i$ and $j \ge P_b$. No query of row $b$ may ever attend to a padding
   column of row $b$ other than itself, in this call or in any later one.
3. **Padding queries must stay finite.** A query that may attend to nothing has a
   softmax of `0 / 0`. Its NaN output becomes a NaN key in the next layer, and there
   `NaN + (-inf)` is NaN, not `-inf`: the additive mask no longer hides it and every
   row that masks it is destroyed. Let each padding query attend to itself.
4. **Positions.** The real token in column $j$ of row $b$ has position $j - P_b$, so
   every row starts at position 0, as it does alone. The positions of padding slots
   do not matter (they must be valid non-negative integers).
5. **One call per step afterwards**, with one token per row, shape $(B, 1)$, and the
   cache. A running row is fed the token it has just generated, at position
   $L_b + s - 1$ for the $s$-th generated token; it may attend to every column
   except its row's padding. A finished row is fed any token; its logits are ignored
   and nothing is appended to it again.
6. **Stop** as soon as every row is finished. The model is called exactly
   $g_{\max}$ times, so the last token of the longest row is never fed.

![The padded batch of the example and the mask of its second row. Left padding puts every row's last real token in the last column; positions restart at 0 after the padding; padding keys are hidden from every query except themselves.](assets/left-padding.svg)

**Evaluation rules.** The judge observes where your code evaluates (`EvalSpy`):

- `mx.eval` is called at most once per step: at most $g_{\max}$ times in all.
  `mx.async_eval` is not used.
- At most one **read** per step, $g_{\max}$ in all: one `.tolist()` (or `np.array`)
  of the whole batch of tokens, not one `.item()` per row.

**Cost limits.** `model.counters.forward_calls` is at most $g_{\max}$ and
`model.counters.tokens_processed` is at most $B \cdot T + B \cdot (g_{\max} - 1)$.

## Shapes

| name | shape | dtype |
|---|---|---|
| `prompts` | list of $B$ lists of $L_b$ ints | Python `int` |
| `tokens` (to `model`) | `(B, T)`, then `(B, 1)` | integer |
| `mask` (to `model`) | broadcastable to `(B, 1, T, C + T)` | float32 |
| `positions` (to `model`) | `(B, T)`, then `(B, 1)` | integer |
| `logits` (from `model`) | `(B, T, vocab_size)` | float32 |
| result | list of $B$ lists of $g_b$ ints | Python `int` |

## Example

```python
model = MLXTinyLM(seed=1)  # vocab_size 64
prompts = [[5, 17, 42, 9, 3], [8, 1], [30, 31, 32]]

batch_generate(model, prompts, max_new_tokens=6, eos_id=1)
# [[46, 30, 1],
#  [30, 34, 36, 44, 8, 43],
#  [58, 41, 51, 60, 6, 56]]
```

$T = 5$ and the padding counts are $P = (0, 3, 2)$. The first call passes

```
tokens      positions
[[ 5, 17, 42,  9,  3],     [[0, 1, 2, 3, 4],
 [ ., .,  .,   8,  1],      [., ., ., 0, 1],
 [ ., ., 30,  31, 32]]      [., ., 0, 1, 2]]
```

(`.` is a padding slot) and reads the first tokens `46, 30, 58` from the last column.
The second call passes `[[46], [30], [58]]` at positions `[[5], [2], [3]]` with a
mask of shape `(3, 1, 1, 6)` that is `-inf` on columns 0 to 2 of row 1 and on columns
0 and 1 of row 2.

Row 0 generates token 1, the EOS, as its third token and is finished: `[46, 30, 1]`.
Rows 1 and 2 never generate it and run to the limit of 6. The `1` at the end of the
prompt `[8, 1]` is an ordinary token. So $g_{\max} = 6$: six model calls, six
evaluations, six reads, and $3 \cdot 5 + 3 \cdot 5 = 30$ tokens processed. For the
last three steps row 0 is fed a filler.

## Constraints

- $1 \le B \le 8$, $1 \le L_b \le 24$, $1 \le$ `max_new_tokens` $\le 20$.
- The tokens are compared exactly with $\operatorname{solo}$ run without a batch or a
  cache. The test models keep the two largest logits more than `1e-3` apart at every
  step, far more than the difference between a batched and a single pass (about
  `1e-6`); if two logits were equal, the argmax would be the lower token id.
- The tests use models with RoPE and models with absolute sinusoidal positions
  (`rope=False`), with two or three layers.
- The evaluation rules and the cost limits are exact counts. A solution with the
  right tokens that breaks one is rejected.

## Why this matters

Batching is how a GPU's idle capacity becomes throughput: on this machine, 32 prompts
through a small transformer take 0.16 s in one batch and 4.3 s one after another.
Real requests never have equal lengths, so every batched server pads, and padding is
where batched outputs quietly stop matching single-request outputs. The failures are
graded. Padding on the right, or attending to it, changes the answers of every short
row. A mask applied to the prompt but forgotten on the decode steps gets the first
token right and drifts afterwards. A fully masked padding row turns into NaN and
takes the batch with it, which is why `mlx-lm` and `transformers` both special-case
it. And a row that has finished must stop producing output while still occupying its
slot in the rectangle, because removing it would change every shape in the cache.
