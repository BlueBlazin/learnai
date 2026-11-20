Unroll the step rule inside one chunk. Token $i$ writes
$\delta_i = \beta_i\,(v_i - \bar S k_i)$ along $k_i$, and the memory it sees is the
chunk's starting memory, decayed, plus the earlier writes of the same chunk, each
decayed from its position to $i$. Write $\bar S k_i$ that way. Which unknowns appear
on both sides, and what shape does the system have if you order the tokens?

---

The system is $\delta_i + \sum_{j<i} \beta_i (k_i \cdot k_j) G_{ij}\,\delta_j =
\beta_i v_i - \beta_i e^{c_i} S_0 k_i$: unit lower triangular, with a right-hand side
that is linear in $S_0$. Solve it once for the part without $S_0$ ($U$) and once for
the coefficient of $S_0$ ($W$); then $\delta = U - W S_0^\top$ needs the memory once per
chunk, through `memory.read`.

Shape the data so that this happens for every chunk at once: move the head axis to
the front, pad $T$ up to a multiple of `chunk_size`, and reshape to
`(H, chunks, C, ...)`. Padding rows must change nothing: a zero key and a zero value
write nothing, and the log-decay of a padding row has to be 0 (a decay of 1), not a
decay of 0. Drop the padded rows of `Y` at the end.

---

Build $G$ safely: `exp(where(i >= j, c[..., :, None] - c[..., None, :], -inf))`, so
the upper triangle is $e^{-\infty} = 0$ and nothing overflows. Then

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

Without a triangular solver, forward substitution over the $C$ rows works for all
chunks at once: row $i$ of the solution is row $i$ of the right-hand side minus
`A[..., i, :i] @ solution[..., :i, :]`.
