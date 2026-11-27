## The idea

Consider one chunk of $C$ tokens starting from memory $S_0$. Let $\delta_i$ denote the vector that token $i$ writes along $k_i$. Immediately after token $i$, the memory holds the decayed starting state plus every write up to position $i$, each decayed from its own position to $i$:

$$
S^{(i)} = e^{c_i} S_0 + \sum_{j \le i} e^{c_i - c_j}\,\delta_j\,k_j^\top .
$$

Here $c_i = \sum_{r \le i} \texttt{log\_g}_r$ is the cumulative log-decay, so $e^{c_i - c_j} = g_{j+1} \cdots g_i$.

By the step rule, $\delta_i = \beta_i\,(v_i - \bar S k_i)$, where $\bar S$ is the decayed memory just before token $i$'s write. That matrix $\bar S$ is the same sum as $S^{(i)}$ without the $j = i$ term. Substituting $\bar S$ into the definition of $\delta_i$ and moving all $\delta_j$ terms to the left gives

$$
\delta_i + \sum_{j < i} \underbrace{\beta_i\,(k_i \cdot k_j)\,e^{c_i - c_j}}_{A_{ij}}\,\delta_j
= \beta_i\,v_i - \beta_i\,e^{c_i}\,S_0\,k_i .
$$

Stacking the $\delta_i$ vectors as rows of $V'$ puts this in matrix form:

$$
(I + A)\,V' = \operatorname{diag}(\beta)\,V - \operatorname{diag}(\beta e^{c})\,K S_0^\top .
$$

Because a write depends only on earlier writes, $A$ is strictly lower triangular and $I + A$ is unit lower triangular. Crucially, the right-hand side is linear in $S_0$, so the solution splits into a term independent of $S_0$ and a linear coefficient multiplying $S_0^\top$:

$$
V' = U - W S_0^\top .
$$

Both $U$ and $W$ can be computed across all chunks in batched triangular solves before reading any memory. The only sequential work left is one `memory.read` and one `memory.update` per chunk, which the counting `memory` object enforces directly.

The outputs and final state follow from the first equation. Evaluating $y_i = S^{(i)} q_i$ gives $Y = \operatorname{diag}(e^{c})\,Q S_0^\top + R V'$, where $R_{ij} = (q_i \cdot k_j)\,e^{c_i - c_j}$ for $j \le i$. The diagonal is included because each token reads after writing. Finally, the memory after the chunk is $S_1 = S^{(C-1)}$.

The Gated DeltaNet paper calls $I + A$ a "UT transform" and derives the same system from a product of Householder-like matrices $(I - \beta k k^\top)$. Unrolling the recurrence directly leads to the same equations.

## Reference solution (PyTorch)

```python
def gdn_chunked(q, k, v, log_g, beta, memory, chunk_size):
    steps, heads = q.shape[0], q.shape[1]
    pad = -steps % chunk_size

    def chunks(x):  # (T, H, ...) -> (H, chunks, C, ...), zero rows appended
        x = F.pad(x.transpose(0, 1), (0, 0) * (x.ndim - 2) + (0, pad))
        return x.reshape(heads, -1, chunk_size, *x.shape[2:])

    q, k, v, log_g, beta = (chunks(x) for x in (q, k, v, log_g, beta))
    c = torch.cumsum(log_g, dim=-1)
    lower = torch.ones(chunk_size, chunk_size, dtype=torch.bool).tril()
    G = torch.exp((c[..., :, None] - c[..., None, :]).masked_fill(~lower, -torch.inf))

    k_beta, v_beta = k * beta[..., None], v * beta[..., None]
    system = torch.eye(chunk_size) + ((k_beta @ k.transpose(-1, -2)) * G).tril(-1)
    U = torch.linalg.solve_triangular(system, v_beta, upper=False, unitriangular=True)
    W = torch.linalg.solve_triangular(
        system, k_beta * torch.exp(c)[..., None], upper=False, unitriangular=True
    )
    read = ((q @ k.transpose(-1, -2)) * G).tril()
    q_start = q * torch.exp(c)[..., None]
    k_end = k * torch.exp(c[..., -1:] - c)[..., None]
    total = torch.exp(c[..., -1])  # (H, chunks)

    ys = []
    for i in range(q.shape[1]):
        seen = memory.read(torch.cat((W[:, i], q_start[:, i]), dim=1))  # (H, 2C, Dv)
        written = U[:, i] - seen[:, :chunk_size]
        ys.append(seen[:, chunk_size:] + read[:, i] @ written)
        memory = memory.update(total[:, i], written, k_end[:, i])
    return torch.cat(ys, dim=1)[:, :steps].transpose(0, 1), memory
```

Padding the last chunk with zeros keeps every chunk the same shape. A padding row has a zero key and zero value, so $\delta = 0$ regardless of $\beta$. It also has a log-decay of 0 (meaning a decay factor of 1), so the memory passes through the padded positions unchanged.

In JAX, `jax.lax.linalg.triangular_solve` supports batched solves directly. In MLX 0.32.4, `mx.linalg.solve_triangular` must be called with `stream=mx.cpu` (verified on this project's machine).

## Common mistakes

- **Running a per-token loop.** This produces the right numbers but makes one read and one write per token. Compiling it can speed up execution, yet it still performs $T$ sequential steps instead of $n$.
- **Making two reads per chunk.** Reading $W$ and $\operatorname{diag}(e^c) Q$ in separate calls exceeds the limit of one read per chunk. Concatenate the rows into a single `(H, 2C, Dk)` tensor.
- **Copying the memory out.** Reading with identity rows recovers $S$ directly, allowing an arbitrary local loop. That requires $D_k$ rows in one call, which `max_rows` rejects whenever $D_k > 2 \cdot \texttt{chunk\_size}$.
- **Forgetting to decay the carried memory.** Computing $S_1 = S_0 + \dots$ instead of $e^{c_{C-1}} S_0 + \dots$ skips the decay across every chunk boundary.
- **Forgetting to decay writes to the end of the chunk.** Using $V'^\top K$ without the factor $e^{c_{C-1} - c}$ treats every write as if it occurred at the final position of the chunk.
- **Using a strictly lower-triangular read matrix.** Reusing the mask of $A$ for $R$ drops the diagonal, so tokens cannot see their own writes.
- **Leaving $S_0$ out of the writes.** Setting $V' = U$ is exact for a single chunk starting from zero memory, but fails from the second chunk onward.
- **Reading $S_0$ without decay.** Computing $Q S_0^\top$ instead of $\operatorname{diag}(e^{c})\,Q S_0^\top$ ignores how much the starting memory decayed before each token.
- **Applying $\beta$ only to values.** Row $i$ of the system matrix $A$ also requires the factor $\beta_i$.
- **Padding with a decay of 0 instead of a log-decay of 0.** Padding `log_g` with a large negative number leaves `Y` unaffected but erases the final memory. Padding with $-\infty$ puts $\infty - \infty$ into the pairwise differences and produces NaN.
- **Masking after the exponential.** Computing `exp(c_i - c_j) * mask` evaluates $e^{+1000}$ above the diagonal for fast-decaying heads, turning $\infty \cdot 0$ into NaN. Always mask the exponent before calling `exp`.

## Complexity

With $n = \lceil T / C \rceil$ chunks, the algorithm performs $O(n\,C^2\,(D_k + D_v))$ arithmetic in batched matrix products and a batched triangular solve, with $n$ sequential steps instead of $T$. Larger chunks reduce the number of sequential steps while increasing the triangular system size quadratically. In production, the default chunk size is 64. For the inputs in this problem, float32 chunked execution differs from the float64 step loop by less than $10^{-5}$.

Wall-clock time is not judged because it reflects framework overhead more than the algorithm. Measured on an Apple M1 Max at $T = 2048$, $H = 4$, $D_k = D_v = 16$, and chunk size 64 with `memory` as a plain tensor, an uncompiled Python token loop is 15x slower in PyTorch and 80x slower in JAX than the chunked form. On CPU, the same token loop under `jax.jit` with `jax.lax.scan` is faster than the uncompiled chunked form. The chunked form pays off by exposing parallel matrix work per sequential step on wide accelerators.

## In the wild

Checked on 2026-11-21 by reading and running `transformers` 5.23.0.

- `torch_chunk_gated_delta_rule` in `transformers/models/qwen3_5/modeling_qwen3_5.py` implements this exact algorithm with the state transposed. It pads with zeros, masks the pairwise log-decay to `-inf` before calling `exp`, calls `torch.linalg.solve_triangular(..., unitriangular=True)` twice (`new_values` is $U$ and `k_cumdecay` is $W$), and scans over chunks. During model export, it builds the inverse via forward substitution over chunk positions instead, because not every export target supports triangular solves. Two tests compare your function against its outputs for chunk sizes 16 and 4 (`fixtures/make.py` regenerates them).
- `Qwen3_5GatedDeltaNet.forward` calls the chunked form for every input except a single token with a cached state. Prompts are prefilled in chunks, and autoregressive decoding runs token by token. The library function also L2-normalizes keys and queries and scales queries, whereas here the inputs arrive already normalized.
- The chunkwise derivation appears in section 3.3 of "Gated Delta Networks: Improving Mamba2 with Delta Rule" (arXiv 2412.06464, ICLR 2025).

Deliberate simplifications: there is no batch axis, inputs arrive in float32 (the library casts internally), test chunks have size at most 16, and the memory is wrapped in a counting object rather than passed as a plain tensor.
