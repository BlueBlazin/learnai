Unroll the step rule inside a single chunk. Token $i$ writes $\delta_i = \beta_i\,(v_i - \bar S k_i)$ along $k_i$, and the memory $\bar S$ it sees is the chunk's starting memory (decayed to position $i$) plus all earlier writes in the same chunk (each decayed from its position to $i$). Substitute that expression for $\bar S k_i$ into $\delta_i$. Which unknowns appear on both sides, and what structure does the resulting linear system have when ordered by token position?

---

The system is $\delta_i + \sum_{j < i} \beta_i (k_i \cdot k_j) G_{ij}\,\delta_j = \beta_i v_i - \beta_i e^{c_i} S_0 k_i$. It is unit lower triangular, and its right-hand side is linear in $S_0$. Solve it once for the $S_0$-independent part ($U$) and once for the coefficient of $S_0$ ($W$). Then $\delta = U - W S_0^\top$ needs `memory.read` only once per chunk.

Reshape the inputs so that the solve runs across all chunks at once. Move the head axis to the front, pad $T$ up to a multiple of `chunk_size`, and reshape to `(H, chunks, C, ...)`. Padding rows must leave the state unchanged, so pad keys and values with 0 and pad `log_g` with 0 (which gives a decay factor of 1, not 0). Slice off the padded rows of `Y` at the end.

---

Build $G$ safely with `exp(where(i >= j, c[..., :, None] - c[..., None, :], -inf))`, so the upper triangle evaluates to $e^{-\infty} = 0$ without overflowing. Then compute the chunks and scan the memory:

```python
system = eye(C) + tril((k_beta @ k.mT) * G, -1)
U = solve_triangular(system, v_beta)                    # lower, unit diagonal
W = solve_triangular(system, k_beta * exp(c)[..., None])
read = tril((q @ k.mT) * G)                             # diagonal included
q_start, k_end = q * exp(c)[..., None], k * exp(c[..., -1:] - c)[..., None]
for i in range(n_chunks):                               # one read, one update
    seen = memory.read(concatenate((W[:, i], q_start[:, i]), axis=1))   # (H, 2C, Dv)
    written = U[:, i] - seen[:, :C]
    ys.append(seen[:, C:] + read[:, i] @ written)
    memory = memory.update(exp(c[:, i, -1]), written, k_end[:, i])
```

If you prefer not to use a triangular solver, forward substitution over the $C$ rows works across all chunks at once. Row $i$ of the solution is row $i$ of the right-hand side minus `A[..., i, :i] @ solution[..., :i, :]`.
