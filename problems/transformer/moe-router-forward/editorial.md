## The idea

An MoE layer replaces the block's single MLP with $E$ of them and a gate. For each
token the gate (the router) produces a sparse weight vector $w \in \mathbb{R}^E$ with
exactly $k$ nonzero entries that sum to 1, and the layer's output is $\sum_e w_e f_e(x)$.
Because $w$ is sparse, only $k$ of the $E$ experts need to run for that token. That is
the whole point: the model owns $E$ MLPs' worth of parameters and pays for $k$.

The weight vector is built in three steps, per token: softmax over all $E$ logits,
keep the top $k$, divide by their sum. There is a useful identity hiding in it. For a
selected expert $e \in S$,

$$
w_e = \frac{p_e}{\sum_{j \in S} p_j}
    = \frac{e^{r_e} / Z}{\sum_{j \in S} e^{r_j} / Z}
    = \frac{e^{r_e}}{\sum_{j \in S} e^{r_j}} ,
$$

since the partition function $Z$ of the full softmax cancels. So "softmax over all
experts, top-k, renormalize" is exactly "top-k of the logits, then softmax over the
selected logits". The two orders are the same function, and either is a correct
solution. They stop being the same only if a step is done wrong: skipping the
renormalization in the first form, or filling the unselected logits with something
other than $-\infty$ in the second.

Selection is the one discontinuous step. A token whose $k$-th and $(k+1)$-th logits
are equal must still get a definite answer, hence the tie rule. `topk` in the three
frameworks does not document which of several equal values it returns, so the
reference ranks with a stable sort.

## Reference solution (PyTorch)

```python
def moe(x, w_router, experts_w, k):
    w_gate, w_up, w_down = experts_w
    logits = x @ w_router.T  # (B, T, E)
    probs = ops.softmax(logits)
    order = torch.argsort(-logits, dim=-1, stable=True)
    rank = torch.argsort(order, dim=-1)
    weights = torch.where(rank < k, probs, 0.0)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    y = torch.zeros_like(x)
    for e in range(w_router.shape[0]):
        y = y + weights[..., e : e + 1] * ops.swiglu(x, w_gate[e], w_up[e], w_down[e])
    return y
```

`argsort` of the negated logits lists each token's experts from best to worst, and a
stable sort lists equal logits in index order. The second `argsort` inverts that
permutation, giving each expert's rank, so `rank < k` is the selection as a dense
mask. The top expert always has probability at least $1/E$, so the sum in the
renormalization is never zero.

With the kit, the equivalent second form is one line, because `ops.top_k_filter` uses
the same tie rule and marks removed entries with $-\infty$:

```python
weights = ops.softmax(ops.top_k_filter(logits, k))
```

The loop over experts could be a single `einsum` over the stacked weights; a loop is
clearer and is what the sparse version will need anyway.

## Common mistakes

- **No renormalization.** Using the top-k probabilities as they come out of the
  softmax gives weights that sum to less than 1, by a different amount for every
  token: a token whose router is unsure gets a smaller MLP update than one whose
  router is confident. With identical experts the output should equal that one
  expert's output; without renormalization it is scaled down.
- **Masking with 0 instead of $-\infty$.** In the "top-k, then softmax" form, an
  unselected logit set to 0 still gets weight $e^0 / \dots$, which is not small at
  all when the selected logits are negative. Only $-\infty$ removes an entry from a
  softmax.
- **Routing per sequence.** Pooling the sequence (or taking its first token) and
  routing once gives weights of shape `(B, 1, E)` that broadcast without complaint.
  Routing is per token: neighboring tokens routinely use different experts.
- **The softmax on the wrong axis.** Logits have shape `(B, T, E)`. A softmax over
  axis 1 normalizes each expert across the tokens of a sequence. When $T = E$ nothing
  about the shapes gives it away.
- **Uniform weights.** Averaging the $k$ selected experts ignores how strongly the
  router preferred each one. For $k = 1$ it happens to be right.
- **Ties broken the other way.** An ascending sort followed by "take the last $k$"
  prefers the highest index among equals. Exact ties are rare with trained float
  weights, but a zero or duplicated router row is enough to produce them, and an
  engine whose outputs depend on an unspecified sort order is not reproducible.

## Complexity

Dense evaluation costs $O(B\,T\,E\,D\,H)$ for the experts, plus $O(B\,T\,E\,D)$ for
the router and $O(B\,T\,E \log E)$ for the sort: $E / k$ times more expert arithmetic
than necessary. The sparse version runs each expert on its own tokens only, for
$O(B\,T\,k\,D\,H)$, the same cost as $k$ ordinary MLPs.

## In the wild

Hugging Face's `MixtralSparseMoeBlock` is this specification in the same order: a
bias-free linear `gate`, `F.softmax` over the experts, `torch.topk` with
`top_k = 2`, division of the kept weights by their sum, and then a loop over the
experts in which each one runs on the tokens that selected it and the result is
scattered back with `index_add_`. Mixtral's experts are SwiGLU MLPs like the ones here
(its `w1`, `w3`, `w2` are gate, up, and down).

The routing rule is a property of the model family, and not all families use this
one. DeepSeek's MoE models add shared experts that every token uses, and score routed
experts differently; some models skip the renormalization by design. As with norm
placement and RoPE layout, the only safe source is the reference implementation of
the model you are loading.

MoE changes the serving economics more than the math. All $E$ experts must be in
memory although each token reads $k$ of them, so memory, not compute, is the binding
constraint, and within a batch each expert sees a different, unpredictable number of
tokens, which makes the per-expert batches ragged.
