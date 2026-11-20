## The idea

A linear-attention layer replaces the growing list of keys and values with one matrix
per head. The simplest such memory is additive: $S \leftarrow S + v\,k^\top$, read with
$y = S q = \sum_i v_i\,(k_i \cdot q)$. It never forgets and never corrects: two tokens
with the same key leave the sum of their values under it.

The delta rule changes the write. Ask the memory what it returns for the new key,
$S k$, and store only the error:

$$
S' = S + \beta\,(v - S k)\,k^\top = S\,(I - \beta\,k k^\top) + \beta\,v\,k^\top .
$$

The second form shows what happens: the component of the memory along $k$ is shrunk by
$1 - \beta\,\lVert k \rVert^2$ and replaced by $\beta v$; everything orthogonal to $k$
is untouched, because $(I - \beta k k^\top) k' = k'$ when $k \cdot k' = 0$. With a
unit key and $\beta = 1$ the old content under $k$ is erased completely and
$S' k = v$. This is one step of gradient descent on $\tfrac12 \lVert S k - v \rVert^2$
with step size $\beta$, which is where the name comes from.

The gate adds forgetting: the whole memory is multiplied by $g \in (0, 1)$ first. The
complete recurrence is the paper's Eq. 10,

$$
S_t = S_{t-1}\,\bigl(g_t\,(I - \beta_t\,k_t k_t^\top)\bigr) + \beta_t\,v_t\,k_t^\top,
\qquad y_t = S_t\,q_t ,
$$

and expanding it gives the three steps of the statement: decay, then a correction
computed from the *decayed* memory, then a read from the *updated* one.

The parameterization of $g$ is Mamba-2's. $e^{A_{\log}}$ is a positive rate per head
and $\operatorname{softplus}(a + \texttt{dt\_bias})$ a positive step size per token, so
the exponent is negative and $g$ cannot exceed 1. A head with a large rate forgets
within a few tokens, one with a small rate keeps its memory for thousands.

## Reference solution (PyTorch)

```python
def gdn_gates(a, b, A_log, dt_bias):
    g = torch.exp(-torch.exp(A_log) * F.softplus(a + dt_bias))
    return g, torch.sigmoid(b)


def gdn_step(S, q, k, v, g, beta):
    S = S * g[:, None, None]  # decay
    predicted = torch.einsum("hvk,hk->hv", S, k)  # what the memory holds under k
    delta = beta[:, None] * (v - predicted)
    S = S + delta[:, :, None] * k[:, None, :]  # write the correction along k
    y = torch.einsum("hvk,hk->hv", S, q)  # read after the write
    return y, S
```

The state layout `(H, Dv, Dk)` is the one of `mlx-lm`. `transformers` stores the
transpose, `(H, Dk, Dv)`, and writes `k.unsqueeze(-1) * delta.unsqueeze(-2)`; the
numbers are the same.

## Common mistakes

- **Reading before the write.** $y = \bar S q$ is the delta rule of a model in which a
  token cannot see its own value. With an empty memory it returns zeros for the first
  token.
- **The additive write**, $\beta\,v\,k^\top$ with no $-\bar S k$. It agrees with the
  delta rule on an empty memory and nowhere else, so a test on the first token alone
  does not catch it.
- **Decay after the write.** The value just written is multiplied by $g$ before it is
  read, and $S' k = g\,v$ instead of $v$. It is invisible when $g = 1$.
- **The correction from the undecayed memory**, $v - S k$ added to $g S$. The memory
  then returns $v - (1 - g)\,S k$ under $k$. Also invisible when $g = 1$, and on an
  empty memory.
- **$\beta$ on the value only**, $(\beta v - \bar S k)\,k^\top$. With $\beta = 1$ it is
  correct; with $\beta = 0$ it erases what was stored under $k$ instead of leaving the
  memory alone.
- **No softplus in the gate.** $\exp(-e^{A_{\log}}(a + \texttt{dt\_bias}))$ is greater
  than 1 for a negative argument: the memory grows by that factor on every such token.
- **`dt_bias` outside the softplus.** `softplus(a) + dt_bias` can also be negative.
- **Normalizing the key inside the step.** The layer does normalize keys, once, before
  the recurrence; a step that does it again is correct for unit keys only, and a
  caller that passes scaled keys on purpose gets a different model.
- **Updating the memory in place.** `S *= g` changes the tensor the caller still
  holds. The outputs are right; a caller that continues twice from one state is not.
- **Calling the kit's oracle.** It is the answer, in numpy.

## Complexity

$O(H\,D_v\,D_k)$ time per token and the same memory, independent of how many tokens
came before. A layer with 16 heads of $128 \times 128$ (Qwen3.5-0.8B) carries 262,144
numbers of state. One of that model's own attention layers caches 2 key-value heads
of width 256, 1,024 numbers per token for keys and values together, so the memory
equals the cache of 256 tokens and stays that size at any context length.

## In the wild

Checked on 2026-11-21 by reading and running the code installed with `transformers`
5.23.0 and by reading `mlx-lm` 0.32.2.

- **The recurrence.** `torch_recurrent_gated_delta_rule` in
  `transformers/models/qwen3_5/modeling_qwen3_5.py` runs exactly these steps per token:
  multiply the state by `decay.exp()`, compute `kv_mem`, form
  `delta = (v_t - kv_mem) * beta_t`, add the outer product, and read with the query.
  The hidden test "six tokens in a row" compares your functions with its output
  (`fixtures/make.py` regenerates the arrays). The Metal kernel in
  `mlx_lm/models/gated_delta.py` performs the same three updates per token.
- **The gates.** `Qwen3_5GatedDeltaNet.forward` computes
  `g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)` and
  `beta = b.sigmoid()`, where `a` and `b` are linear projections of the layer input.
  It keeps the decay in log space and exponentiates inside the recurrence; this
  problem returns $g$ itself.
- **What the library does around the step,** and this problem leaves to the layer
  (`gated-delta-layer`): L2-normalizing queries and keys, multiplying the query by
  $D_k^{-1/2}$, and repeating key heads over value heads. In `transformers` those three
  happen inside the same function.
- **One token or many.** `transformers` calls the per-token function only for a single
  cached token; a call with several tokens goes through a chunked form that gives the
  same result (`gated-delta-chunked`).
- **The paper.** "Gated Delta Networks: Improving Mamba2 with Delta Rule"
  (arXiv 2412.06464, ICLR 2025), Eq. 10, is the recurrence above with $\alpha_t$ for
  $g_t$.
