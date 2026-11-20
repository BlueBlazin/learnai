Implement a mixture-of-experts layer whose routed experts work at a fraction of the
model's width: project down once, run and mix the chosen experts there, normalize the
mixture, and project back up, while the router and the shared expert stay at full
width. The expert work is counted.

## Specification

```python
def latent_moe(x, w_router, w_down, w_up, norm_gain, experts, shared_w, k): ...
def expert_params(hidden, latent, inter, n_experts): ...  # -> (with_latent, without_latent)
```

There are $N$ tokens of width $D$ and $E$ routed experts of latent width $L < D$. For
**each token** $x \in \mathbb{R}^D$ independently:

1. **Route at full width.** $s = \sigma(W_{\text{router}}\, x) \in \mathbb{R}^E$. The
   chosen set $S$ is the $k$ experts with the largest scores; equal scores are ranked
   by index, lowest first. The weights are

$$
w_e = \frac{s_e}{\sum_{j \in S} s_j + 10^{-20}} \qquad (e \in S) .
$$

   This is the router of `sigmoid-router-shared-expert` with no bias and scale 1.

2. **Project down, once:** $z = W_{\text{down}}\, x \in \mathbb{R}^L$.
3. **Mix at latent width:** $m = \sum_{e \in S} w_e\, f_e(z)$, where $f_e$ is what
   `experts(e, ·)` computes on one row.
4. **Normalize the mixture** over its $L$ channels, with the gain $g$ = `norm_gain`:

$$
n = g \odot \frac{m}{\sqrt{\tfrac{1}{L}\sum_{i} m_i^2 + 10^{-5}}} ,
$$

   which is `ops.rms_norm(m, norm_gain, eps=1e-5)` from the kit.
5. **Project up and add the shared expert**, which reads the original input:
   $y = W_{\text{up}}\, n + \texttt{ops.swiglu}(x, \texttt{*shared\_w})$.

Arguments and result of `latent_moe`:

- `x`: float32, shape `(N, D)`.
- `w_router`: `(E, D)`; `w_down`: `(L, D)`; `w_up`: `(D, L)`. Float32, in `(out, in)`
  layout, applied as `x @ w.T`. `norm_gain`: float32, shape `(L,)`.
- `experts`: a [`CountingExperts`](../../../python/src/learnai/kit/moe.py) object of
  width $L$. `experts(e, rows)` runs expert `e` (a Python int) on a float32 tensor of
  shape `(n, L)` and returns `(n, L)`, row for row; `experts.n_experts` is $E$. It is
  the only way to evaluate a routed expert. It counts one `expert_calls` per call and
  `n` `expert_rows`, and raises `ValueError` for rows of another width.
- `shared_w`: a tuple `(w_gate, w_up, w_down)` of shapes `(H_s, D)`, `(H_s, D)`,
  `(D, H_s)`.
- `k`: a Python int with $2 \le k \le E$.
- Returns float32, shape `(N, D)`.

**Work.** An expert that at least one token chose is called exactly once, with the
latent rows of exactly the tokens that chose it, in any order. An expert that no token
chose is not called, not even with zero rows. In counters: `expert_rows` equals
$k \cdot N$, and `expert_calls` equals the number of experts chosen by some token.

**`expert_params(hidden, latent, inter, n_experts)`** returns two Python ints: the
number of parameters of the routed experts with the latent projection, and without
it. A routed expert at width $w$ is a SwiGLU MLP with three bias-free matrices: gate
and up of shape $(\texttt{inter}, w)$, down of shape $(w, \texttt{inter})$.

- With: `n_experts` experts at width `latent`, plus $W_{\text{down}}$ and
  $W_{\text{up}}$.
- Without: `n_experts` experts at width `hidden`.

The router, the shared expert, and the gain of the norm are in neither count.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(N, D)` | float32 |
| `w_router` | `(E, D)` | float32 |
| `w_down` | `(L, D)` | float32 |
| `w_up` | `(D, L)` | float32 |
| `norm_gain` | `(L,)` | float32 |
| `rows` passed to `experts(e, rows)` | `(n, L)` | float32 |
| `shared_w` | `(H_s, D)`, `(H_s, D)`, `(D, H_s)` | float32 |
| result of `latent_moe` | `(N, D)` | float32 |
| result of `expert_params` | two Python ints | |

## Example

The sizes of Kimi K3.1's routed experts: model width 7168, latent width 3584, expert
inner width 3072, 896 experts.

```python
expert_params(7168, 3584, 3072, 896)  # (29646389248, 59190018048)
```

Without the latent projection an expert has $3 \cdot 7168 \cdot 3072 = 66{,}060{,}288$
parameters, and 896 of them have 59.19 billion. At latent width an expert has
$3 \cdot 3584 \cdot 3072 = 33{,}030{,}144$, in total 29.60 billion, and the two
projections add $2 \cdot 7168 \cdot 3584 = 51{,}380{,}224$ once: 29.65 billion.

A small one: `expert_params(4, 2, 3, 5)` is `(106, 180)`, from
$5 \cdot 3 \cdot 2 \cdot 3 + 2 \cdot 4 \cdot 2$ and $5 \cdot 3 \cdot 4 \cdot 3$.

For the layer with 12 tokens, $k = 2$, and 8 experts, a correct solution makes at most
8 calls with 24 rows of width $L$ in total. Evaluating every expert on every token
makes 8 calls with 96 rows.

## Constraints

- $1 \le N \le 12$, $D = 16$, $L = 8$, $2 \le E \le 8$, $2 \le k \le E$.
- The output is compared with the dense evaluation of the formulas above to a relative
  tolerance of `1e-4` and an absolute tolerance of `1e-5`. An expert's output for a
  row does not depend on which other rows are in the call, except in the last bits.
- **Cost, checked in every case:** `expert_rows` is exactly $k \cdot N$, and
  `expert_calls` is at most the number of experts that some token chose.
- In every test, the $k$-th and $(k+1)$-th largest scores of a token are either
  exactly equal (inputs made of small integers) or differ by more than $10^{-3}$.
- `expert_params` is compared exactly and must return Python ints.
- A Python loop over experts is expected. No loop over tokens is needed.

## Why this matters

The experts hold most of the parameters of a large mixture-of-experts model: 896
SwiGLU experts at width 7168 would be 59 billion per layer, and running them at half
width halves that for the price of two shared projections. Every tensor in the wrong
versions still has a plausible shape. A norm applied to each expert's output instead
of the mixture, or left out because another model's latent experts have none, runs
without error and returns a residual update of the wrong size in every layer.
