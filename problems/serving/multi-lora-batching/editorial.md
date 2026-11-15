## The idea

`lora-merge` ended with two ways of running an adapted layer: merged,
$x\,(W + \tfrac{\alpha}{r} B A)^{\top}$, and unmerged,
$x W^{\top} + \tfrac{\alpha}{r} (x A^{\top}) B^{\top}$. For one adapter they are the
same numbers, and merged is cheaper. For a batch that mixes adapters only the
unmerged form works, because it separates the computation into

- a term that is identical for every row whatever its adapter, $x W^{\top}$, which is
  also the expensive one ($\text{in} \times \text{out}$ per row), and
- a term that differs per row and is cheap ($r\,(\text{in} + \text{out})$ per row).

The first term is one call of the base layer on the whole batch. The second is made
uniform by a gather: stack all adapters, index the stack with `adapter_ids`, and
every row gets its own $A$ and $B$ along a new leading axis. A batched matmul then
treats row $b$ with `a[b]` and `b[b]`, which is exactly "each row with its own
adapter" without a loop.

Two details make the gather correct.

**Ranks.** A stack needs one shape, so adapters are zero-padded to $r_\text{max}$.
In the product the padding is harmless: the padded rows of $A$ give zero
coordinates, and they multiply zero columns of $B$. But LoRA's scale is
$\alpha / r$ with the adapter's own rank, so the scale is gathered too,
`alpha / ranks[index]`, one number per row.

**No adapter.** Rows of the base model carry the id $-1$. Negative indices are legal
in all three frameworks and select from the end, so the id cannot be used as an
index. The branch-free fix is to index with a clamped id (any valid adapter) and
multiply that row's update by 0: `where(adapter_ids >= 0, alpha / ranks[index], 0)`.
The base output of such a row is untouched, not multiplied by anything.

## Reference solution (PyTorch)

```python
def batched_lora_forward(x, base, a_stack, b_stack, ranks, adapter_ids, alpha):
    y = base(x)  # once, whole batch
    has_adapter = adapter_ids >= 0
    index = torch.clamp(adapter_ids, min=0)
    a = a_stack[index]  # (B, r_max, in)
    b = b_stack[index]  # (B, out, r_max)
    scale = torch.where(has_adapter, alpha / ranks[index], 0.0)  # (B,)
    tokens = x if x.dim() == 3 else x[:, None, :]  # (B, T, in)
    low = tokens @ a.transpose(1, 2)  # (B, T, r_max)
    delta = (low @ b.transpose(1, 2)) * scale[:, None, None]
    return y + delta.reshape(y.shape)
```

The JAX and MLX versions use `maximum(adapter_ids, 0)`, `where`, and `swapaxes`, and
are otherwise the same lines.

## Common mistakes

- **A Python loop over the rows.** Correct, and it turns one base matmul into $B$
  small ones and $2B$ adapter matmuls with interpreter overhead between them: 6
  base calls instead of 1 in the cost case, 16x to several hundred times slower in
  the speed case.
- **Looping over rows only for the updates.** One base call, so the cost case
  passes, and the per-row overhead is still there. The speed case is for this.
- **Splitting the batch by adapter and calling the base layer per group.** This is
  what "one batch per adapter" looks like inside a single function. The updates are
  right; the shared matmul has been cut into as many pieces as there are adapters,
  which is the cost the whole technique exists to avoid. (Looping over adapters
  for the *updates* only, after one base call, is a legitimate alternative to the
  gather; see below.)
- **Indexing with $-1$.** `a_stack[adapter_ids]` runs without error and gives
  base-model rows the last adapter in the stack. Nothing crashes; a customer of the
  plain model gets somebody's fine-tune.
- **Masking the output instead of the update.** `(y + delta) * mask` zeroes the
  base-model rows.
- **Scaling by the padded rank.** `alpha / a_stack.shape[1]` is right only for the
  adapters of the largest rank; a rank-1 adapter in a stack padded to 8 is applied at
  an eighth of its strength.
- **Using one adapter for the batch.** Code written for a single adapter reads the
  first id and applies it to everyone.
- **Broadcasting the scale on the wrong axis.** With `x` of shape `(B, T, in)`, a
  scale of shape `(B, 1)` lines up with `(T, out)`: an error when $B \ne T$, and
  silently wrong values when they happen to be equal.

## Complexity

One base call, $O(B\,T \cdot \text{in} \cdot \text{out})$, plus
$O(B\,T\,r_\text{max}\,(\text{in} + \text{out}))$ for the updates. The gather copies
$B\,r_\text{max}\,(\text{in} + \text{out})$ numbers: every row carries a private copy
of its adapter, padded to the largest rank. That is the price of the simple
formulation, and the two refinements used in practice remove it: group the rows by
adapter and apply each adapter once to its segment (no copies, no padding; one
small matmul per adapter present), or use a kernel that reads the adapter weights
through the index directly.

## In the wild

- **Punica** ("Punica: Multi-Tenant LoRA Serving", arXiv 2310.18547) introduced the
  batched operator for this, Segmented Gather Matrix-Vector multiplication (SGMV):
  a CUDA kernel that computes the $x A^{\top} B^{\top}$ terms for a batch of requests
  with different adapters, grouping the requests that share one, while a single copy
  of the backbone serves all of them.
- **S-LoRA** ("S-LoRA: Serving Thousands of Concurrent LoRA Adapters", arXiv
  2311.03285) makes the same split, batching the base computation across all
  requests and computing the adapter part separately, argues against merging for
  multi-adapter serving, and adds custom kernels for batches whose adapters have
  different ranks so that the padding of this problem is not paid, plus a unified
  memory pool for adapter weights and KV cache.
- **vLLM** serves adapters per request (`--lora-modules name=path`, selected by the
  request's `model` field) and, in its documentation's words, can "combine base
  model and adapter requests in the same batch". Its LoRA layers keep the adapters
  in stacked tensors and carry a per-token index into them
  (`vllm/lora/punica_wrapper`); the documented semantics of `add_lora_linear` are
  the formula of this problem, `y[i] += x[i] @ lora_a_stacked[indices[i]] @
  lora_b_stacked[indices[i]] * scale`, and the index convention is the same: "An
  index of -1 means no lora should be applied." The index there is per token of the
  flattened batch, which is the `(B, in)` case with $B$ in the hundreds, the shape
  of the speed case.
- **Hugging Face PEFT** supports mixed batches at inference through an
  `adapter_names` argument to the forward pass, one adapter name per row of the
  batch, with a reserved name for rows that use the base model.
