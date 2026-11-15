## The idea

The dense layer of `moe-router-forward` computes $E$ expert outputs per token and
multiplies $E - k$ of them by zero. The sparse layer computes only the $k$ that
count. Its output is the same sum,

$$
y_t = \sum_{e \in S_t} w_{t,e}\, f_e(x_t),
$$

reorganized by expert instead of by token: with $T_e$ the tokens that selected
expert $e$,

$$
y = \sum_{e\,:\,T_e \ne \emptyset} \operatorname{scatter}_{T_e}\!\big( w_{T_e, e} \odot f_e(x_{T_e}) \big).
$$

Each term is three operations. **Gather** the rows $x_{T_e}$ into a dense
$(|T_e|, D)$ batch. **Run** expert $e$ on it, once: this is the only expensive step,
and it is a few large matrix products. **Scatter-add** the weighted result to rows
$T_e$ of $y$. The row counts add up to $\sum_e |T_e| = kN$, because every token is in
exactly $k$ of the sets.

What remains is building the sets $T_e$ without a loop over tokens. The routing is a
table `chosen` of shape $(N, k)$. Flatten it to $kN$ assignments
(token, expert, weight) and sort the assignments by expert. In sorted order each
expert owns one contiguous run, whose length is the number of times it occurs. That
is a counting sort, and it is how the groups are formed in practice: it uses only
`argsort`, a `bincount`, and integer indexing, with no boolean-mask read and no
data-dependent shape inside a tensor operation. The run lengths are read back to
Python, because they are the shapes of the calls.

## Reference solution (PyTorch)

```python
def moe_sparse(x, w_router, experts, k):
    n_tokens = x.shape[0]
    logits = x @ w_router.T
    chosen = torch.argsort(-logits, dim=-1, stable=True)[:, :k]
    weights = torch.softmax(torch.gather(logits, 1, chosen), dim=-1)

    flat_expert = chosen.reshape(-1)
    flat_token = torch.arange(n_tokens).repeat_interleave(k)
    flat_weight = weights.reshape(-1)
    order = torch.argsort(flat_expert, stable=True)
    counts = torch.bincount(flat_expert, minlength=experts.n_experts).tolist()

    y = torch.zeros_like(x)
    start = 0
    for expert, count in enumerate(counts):
        if count == 0:
            continue
        run = order[start : start + count]
        start += count
        tokens = flat_token[run]
        out = experts(expert, x[tokens])
        y = y.index_add(0, tokens, flat_weight[run, None] * out)
    return y
```

- **Weights.** $p_e / \sum_{j \in S} p_j$ is the softmax of the kept logits alone:
  the normalizer over all experts cancels. No full softmax is needed.
- **Alignment.** `tokens`, `flat_weight[run]`, and the rows of `out` are all indexed
  by position in the run, so row $j$ of the expert's output meets the weight and the
  destination of the same assignment. Any permutation of the run would do, as long
  as all three use it.
- **Accumulation.** Each row of `y` is written $k$ times, once per expert of that
  token. Within one expert's run the token indices are distinct, so an indexed `+=`
  is also correct here; across runs the additions accumulate.

The JAX and MLX references differ in two lines: the counts are computed with
`np.bincount` on the host, and the scatter is `y.at[tokens].add(...)`.

## Common mistakes

- **Dense evaluation.** The right numbers at $E / k$ times the cost: 4 times for
  Mixtral, 16 times for Qwen3-30B-A3B.
- **One call per token.** `for t in range(N): experts(e, x[t:t+1])` has the right
  row count and $kN$ calls. On real hardware each call is a kernel launch and a
  matrix product too small to use the device; throughput collapses.
- **One group per slot.** Grouping the first choices and the second choices
  separately is natural when the routing is a `(values, indices)` pair from `topk`.
  An expert that is first choice for some tokens and second choice for others is
  then called twice. Flatten the slots before grouping.
- **Calling idle experts.** A loop `for e in range(E)` that gathers and calls
  unconditionally issues empty calls. With 128 experts and a batch of a few tokens,
  most calls are empty.
- **A contiguous scatter.** Writing an expert's output to `y[first : first + n]`
  assumes its tokens are adjacent. That is true in toy tests where neighboring
  tokens are similar, and false in any real batch.
- **Assignment instead of accumulation.** `y[tokens] = ...` keeps only the last
  expert of each token. With $k = 1$ it is correct, which is how it survives.
- **Forgetting to renormalize.** Gathering the kept entries of the full softmax
  gives weights that sum to less than 1 and shrinks every MoE update.

## Complexity

One router product, $O(NE)$; a sort of $kN$ small integers; and expert work
proportional to $kN$ rows instead of $EN$. The number of expert calls is at most
$\min(E, kN)$. Memory for the gathered rows is $O(kND)$ in total, one group at a
time.

## In the wild

Hugging Face `transformers` runs Mixtral's block this way
(`MixtralSparseMoeBlock.forward`, also the Qwen3 MoE block): it takes `topk` of the
router's softmax, renormalizes, builds a one-hot mask of the selections, computes
`expert_hit`, the experts that received at least one token, and loops over those
only; for each it finds its tokens with `torch.where`, runs the expert on
`hidden_states[top_x]`, and accumulates with `index_add_`.

Fused implementations go further and never materialize per-expert batches. vLLM's
fused MoE first aligns the work by expert (`moe_align_block_size` produces the token
ids sorted by expert index, padded to the kernel's block size) and runs one grouped
matrix product over the sorted rows. mlx-lm's `SwitchGLU` gathers each token's
expert weights directly, and when a call has 64 or more assignments it sorts them by
expert index first and unsorts the result afterwards, for memory locality. Both sort
for the reason the reference does: contiguous runs per expert. The grouping is also
the first step of expert parallelism, where each run is sent to the device that
holds its expert.
