## The idea

Follow one chunk of $C$ tokens from the memory $S_0$. Write $\delta_i$ for the vector
that token $i$ writes along $k_i$, so the memory right after token $i$ is

$$
S^{(i)} = e^{c_i} S_0 + \sum_{j \le i} e^{c_i - c_j}\,\delta_j\,k_j^\top ,
$$

the starting memory and every write so far, each decayed from where it happened to
position $i$ ($c_i$ is the cumulative log-decay, so $e^{c_i - c_j} = g_{j+1} \cdots g_i$).
The step rule defines $\delta_i = \beta_i\,(v_i - \bar S k_i)$ with $\bar S$ the memory
just before the write, which is the same sum without the term $j = i$. Substituting:

$$
\delta_i + \sum_{j < i} \underbrace{\beta_i\,(k_i \cdot k_j)\,e^{c_i - c_j}}_{A_{ij}}\,\delta_j
= \beta_i\,v_i - \beta_i\,e^{c_i}\,S_0\,k_i .
$$

In matrix form, with the $\delta_i$ as rows of $V'$:
$(I + A)\,V' = \operatorname{diag}(\beta) V - \operatorname{diag}(\beta e^{c}) K S_0^\top$.
$A$ is strictly lower triangular because a write depends only on earlier writes. The
right-hand side is linear in $S_0$, so the solve splits into a part that knows nothing
about $S_0$ and a coefficient of it: $V' = U - W S_0^\top$. That split is the whole
trick. $U$ and $W$ can be computed for every chunk of the sequence in one batched
solve before any memory is known, and what remains sequential is one read of the
memory and one write per chunk. This problem makes that visible by handing the memory
over as an object that counts its reads and writes.

The reads and the new memory are the first formula again: $y_i = S^{(i)} q_i$ gives
$Y = \operatorname{diag}(e^{c}) Q S_0^\top + R V'$ with
$R_{ij} = (q_i \cdot k_j) e^{c_i - c_j}$ for $j \le i$, diagonal included because the
read comes after the write; and $S^{(C-1)}$ is $S_1$.

The paper calls $I + A$ the result of a "UT transform" and reaches the same system
through a product of Householder-like factors $(I - \beta k k^\top)$; the derivation
above is the direct one.

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

Padding the last chunk keeps every chunk the same shape. A padding row has a zero
key and value, so $\delta = 0$ whatever $\beta$ is, and a log-decay of 0, so the
memory passes it unchanged. JAX uses `jax.lax.linalg.triangular_solve`, which
batches; MLX 0.32.4 runs `mx.linalg.solve_triangular` only with `stream=mx.cpu`
(measured on this project's machine).

## Common mistakes

- **The per-token loop.** Correct numbers, one read and one write of the memory per
  token. Compiling it changes its speed, not its shape: it is still $T$ dependent
  steps, each waiting for the memory of the one before.
- **Two reads per chunk,** one for $W$ and one for the queries. Stack the rows; the
  limit is one read per chunk.
- **Copying the memory out.** Reading with identity rows returns $S$ itself, after
  which any loop can run on the copy. It needs $D_k$ rows in one call, which
  `max_rows` refuses as soon as keys are wider than two chunks.
- **No decay on the carried memory.** $S_1 = S_0 + \dots$ instead of
  $e^{c_{C-1}} S_0 + \dots$: every chunk boundary forgets to forget.
- **Writes not decayed to the end of the chunk.** $V'^\top K$ without
  $e^{c_{C-1} - c}$ treats every write as if it happened at the last position.
- **A strictly lower read.** Using the mask of $A$ for $R$ drops the diagonal, and no
  token sees its own write.
- **$S_0$ left out of the writes**, $V' = U$. Exact for an empty memory and a single
  chunk, wrong from the second chunk on.
- **$S_0$ read without decay**, $Q S_0^\top$ instead of $\operatorname{diag}(e^{c}) Q S_0^\top$.
- **$\beta$ on the values only.** The system matrix needs $\beta_i$ on row $i$ too.
- **Padding with a decay of 0.** A padding row needs a log-decay of 0. Padding with a
  large negative log-decay (a decay of 0) leaves the outputs right and returns an
  erased memory; padding with $-\infty$ itself also puts $\infty - \infty$ into the
  decays between positions and returns NaN.
- **Masking after the exponential.** `exp(c_i - c_j) * mask` computes $e^{+1000}$
  above the diagonal for a fast-decaying head and turns it into NaN. Mask the
  exponent, or zero the entries with `tril` or `where`.

## Complexity

With $n = \lceil T / C \rceil$ chunks: $O(n\,C^2\,(D_k + D_v))$ arithmetic, all of it in
batched matrix products and one batched triangular solve, and $n$ sequential steps
instead of $T$. Larger chunks mean fewer sequential steps and quadratically larger
systems; the library's default is 64. In float32 the result differs from the step
loop by less than $10^{-5}$ for the inputs of this problem.

Wall-clock time is not judged, because it measures the framework more than the
algorithm. Measured on an Apple M1 Max at $T = 2048$, $H = 4$, widths 16, chunks of
64, with the memory as a plain tensor: an uncompiled Python loop over tokens takes 15
(PyTorch) to 80 (JAX) times as long as the chunked form, while the same loop under
`jax.jit` with `jax.lax.scan` is faster than the uncompiled chunked form on a CPU.
What the chunked form buys is parallel work per sequential step, which matters on
hardware that is wide rather than fast per step.

## In the wild

Checked on 2026-11-21 by reading and running `transformers` 5.23.0.

- `torch_chunk_gated_delta_rule` in
  `transformers/models/qwen3_5/modeling_qwen3_5.py` is this algorithm with the state
  transposed: it pads with zeros, masks the pairwise decay to `-inf` before the
  exponential, calls `torch.linalg.solve_triangular(..., unitriangular=True)` twice
  (`new_values` and `k_cumdecay` are $U$ and $W$), and scans the chunks. When the
  model is being exported it builds the inverse by forward substitution over chunk
  positions instead, because not every export target has the solver. Two tests
  compare your function with its outputs for chunks of 16 and 4 (`fixtures/make.py`
  regenerates them).
- `Qwen3_5GatedDeltaNet.forward` calls it for every input except a single token with
  a cached state, so a prompt is prefilled in chunks and generation proceeds token by
  token. The library function also L2-normalizes and scales the query; here the
  inputs arrive already normalized.
- The chunkwise form is section 3.3 of "Gated Delta Networks: Improving Mamba2 with
  Delta Rule" (arXiv 2412.06464, ICLR 2025).

Deliberate simplifications: no batch axis; float32 inputs (the library casts to
float32 inside the function); chunks of at most 16 in the tests; and the memory as a
counting object, which the library does not have (there it is a tensor).
