Implement one attention step for a transformer layer with a KV cache. Your function appends the keys and values of new tokens to the cached keys and values, then computes causal attention from the new queries over the combined sequence. The output must match full-sequence causal attention regardless of whether the sequence arrives all at once, one token at a time, or in arbitrary chunks.

![This problem is Attention with the KV cache, which appends the new keys and values and reads them all.](assets/map.svg)

## Specification

```python
def attend_step(q_new, k_new, v_new, cache): ...  # returns (out, cache)
```

The input tensors come from an attention layer after its linear projections and head splitting. Their layout is `(B, H, T, D)` for batch size `B`, number of heads `H`, number of new tokens `T` ($T \ge 1$), and head dimension `D`.

Each call receives the queries `q_new`, keys `k_new`, and values `v_new` for `T` new tokens. The `cache` argument holds the keys and values of the `S` earlier tokens as a tuple `(k, v)`, ordered oldest token first. When nothing has been cached yet, `cache` is `None` ($S = 0$). Otherwise, $S \ge 1$.

Compute the step in four stages:

1. **Append.** Concatenate the cached and new keys and values along the sequence axis (axis 2), with cached entries first. When `cache` is `None`, use `k_new` and `v_new` directly.

$$
K = [\,k_{\text{cache}};\ k_{\text{new}}\,], \qquad V = [\,v_{\text{cache}};\ v_{\text{new}}\,],
\qquad \text{both of shape } (B, H, S + T, D).
$$

2. **Score.** For each batch item and head, let $q_i$ be the query vector for new token $i$ ($0 \le i < T$) and let $K_j$ be the key vector at entry $j$ ($0 \le j < S + T$). Compute the scaled dot-product attention scores:

$$
s_{ij} = \frac{q_i \cdot K_j}{\sqrt{D}} .
$$

3. **Mask and normalize.** New token $i$ sits at absolute position $S + i$ in the sequence. It may attend to all positions $j \le S + i$, which includes all $S$ cached tokens, the earlier tokens in the new chunk, and itself. Apply a softmax over those visible positions and set masked positions to zero:

$$
a_{ij} = \begin{cases}
\dfrac{e^{s_{ij}}}{\sum_{l \le S + i} e^{s_{il}}} & j \le S + i, \\[2ex]
0 & j > S + i .
\end{cases}
$$

4. **Mix.** Combine the value vectors $V_j$ using the attention weights:

$$
\text{out}_i = \sum_{j \le S + i} a_{ij}\, V_j .
$$

![Seven tokens over four calls of `attend_step`. The mask has one row per new query and one column per entry of $K$.](assets/cache-steps.svg)

Return a tuple `(out, (K, V))`, where `out` contains the attention outputs for the `T` new tokens and `(K, V)` is the updated cache to pass into the next call. There is no output projection, position encoding, or surrounding model in this problem.

**State in, state out.** Treat `cache` as an immutable value rather than an object with an `append` method. Return a new cache tuple and leave the input tensors unchanged so a caller can reuse an older cache to branch from the same prefix. This design works cleanly across PyTorch, JAX, and MLX because concatenation returns new tensors in all three frameworks.

You may use [`ops.attention`](../../../python/src/learnai/kit/ops/__init__.py) from the kit (`from learnai.kit import ops`), which computes $\operatorname{softmax}(q K^\top / \sqrt{D} + \text{mask})\,V$ for a mask you supply, or write the attention arithmetic yourself.

## Shapes

| name | shape | dtype |
|---|---|---|
| `q_new`, `k_new`, `v_new` | `(B, H, T, D)` | float32 |
| `cache` | `None`, or `(k, v)`, each `(B, H, S, D)` | float32 |
| `out` | `(B, H, T, D)` | float32 |
| returned `k`, `v` | `(B, H, S + T, D)` | float32 |

## Example

Consider one batch row, one head, `D = 2`, and three tokens:

```python
q = [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]
k = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
v = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]  # each of shape (1, 1, 3, 2) once batched
```

Process the sequence as a chunk of two tokens followed by one token:

```python
out, cache = attend_step(q[:, :, :2], k[:, :, :2], v[:, :, :2], None)
# out   = [[[[1.000000, 2.000000],
#            [2.608860, 3.608860]]]]
# cache = (k[:, :, :2], v[:, :, :2])            two entries

out, cache = attend_step(q[:, :, 2:], k[:, :, 2:], v[:, :, 2:], cache)
# out   = [[[[3.510469, 4.510469]]]]
# cache = (k, v)                                three entries
```

- **Step 1** has no cache ($S = 0, T = 2$), so it runs standard causal attention over two tokens. Token 0 sees only itself and outputs $v_0 = [1, 2]$. Token 1 has scores $[0,\ 2/\sqrt{2}] = [0, 1.4142]$ against keys 0 and 1, weights $[0.1956, 0.8044]$, and output $0.1956\,v_0 + 0.8044\,v_1 = [2.6089, 3.6089]$.
- **Step 2** has $S = 2$ cached tokens and $T = 1$ new token. The new token is at position 2 and sees all three entries. It gets scores $[0.7071, 0.7071, 1.4142]$, weights $[0.2483, 0.2483, 0.5035]$, and output $[3.5105, 4.5105]$.

Passing all three tokens in a single call with `cache=None` produces the exact same three output rows and final cache. The test suite verifies this chunk-invariance property across different ways of splitting a sequence.

## Constraints

- $1 \le B, H \le 4$, $1 \le T \le 4$, $0 \le S \le 4$, and $2 \le D \le 16$.
- `out` is compared against both the reference implementation and a single full-sequence causal pass with a relative tolerance of `1e-4` and an absolute tolerance of `1e-5`. Any NaN fails immediately.
- The returned cached keys and values are compared exactly against the concatenated inputs.
- No Python loops over tokens, heads, or batch rows are needed.

## Why this matters

This function is the inner step of every cached decoder. During single-token generation, `T = 1` and the single new query can attend to every entry in the cache, so an implementation that forgets the causal mask still passes decode-only tests.

The rectangular causal mask becomes critical whenever multiple tokens arrive with a non-empty cache. This happens when appending a new chat turn to a cached conversation, prefilling long prompts in chunks to bound latency, or verifying multiple draft tokens in speculative decoding. A buggy mask makes model outputs depend on chunk size or batching decisions, producing silent, hard-to-reproduce regressions.
