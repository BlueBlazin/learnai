## The idea

Each channel of each head runs a linear recurrence with a scalar decay:

$$
h_t = a_t\,h_{t-1} + \Delta_t\,x_t\,B_t^\top, \qquad y_t = h_t\,C_t + D\,x_t,
\qquad a_t = e^{-\Delta_t e^{A_{\log}}} .
$$

It is a discretized differential equation $\dot h = A h + x B^\top$ with
$A = -e^{A_{\log}} < 0$, held for a time $\Delta_t$ that the token chooses: a large
step forgets much and writes much, a small one does neither. The softplus keeps the
step positive, and the clamp keeps it inside a configured range.

Unrolling from $h_{-1} = 0$:

$$
h_t = \sum_{s \le t} \Bigl(\prod_{r = s+1}^{t} a_r\Bigr)\,\Delta_s\,x_s\,B_s^\top
\quad\Longrightarrow\quad
y_t = \sum_{s \le t} \underbrace{(C_t \cdot B_s)\; e^{-e^{A_{\log}} \sum_{r=s+1}^{t} \Delta_r}\; \Delta_s}_{M_{t,s}}\; x_s + D\,x_t .
$$

That is an attention matrix: the score of query $t$ and key $s$ is the dot product
$C_t \cdot B_s$, the "mask" is a decay that depends only on the step sizes between
the two positions, and the values are the inputs scaled by their own step. No softmax,
no learned positions. The product of decays starts at $r = s + 1$ because token $s$
decays the state *before* it writes; its own step does not act on its own input.

The two forms are the two ends of a trade. The matrix costs $O(T^2)$ and is one
batched product: good for prefill, in chunks, with the state carried between chunks.
The step costs $O(1)$ per token and needs only $h$: good for decoding. An engine
needs both, and they are correct only together.

## Reference solution (PyTorch)

```python
def ssd_step(h, x, dt_raw, B, C, A_log, D, dt_bias, dt_limit):
    dt = torch.clamp(F.softplus(dt_raw + dt_bias), min=dt_limit[0], max=dt_limit[1])
    reps = h.shape[0] // B.shape[0]
    B, C = B.repeat_interleave(reps, dim=0), C.repeat_interleave(reps, dim=0)
    decay = torch.exp(-dt * torch.exp(A_log))
    h = h * decay[:, None, None] + (dt[:, None] * x)[:, :, None] * B[:, None, :]
    return torch.einsum("hpn,hn->hp", h, C) + D[:, None] * x, h


def ssd_matrix(dt, B, C, A_log):
    steps, heads = dt.shape
    reps = heads // B.shape[1]
    B, C = B.repeat_interleave(reps, dim=1), C.repeat_interleave(reps, dim=1)
    scores = torch.einsum("thn,shn->hts", C, B)
    c = torch.cumsum(-dt * torch.exp(A_log), dim=0).T  # (H, T)
    lower = torch.ones(steps, steps, dtype=torch.bool).tril()
    decay = torch.exp((c[:, :, None] - c[:, None, :]).masked_fill(~lower, -torch.inf))
    return scores * decay * dt.T[:, None, :]
```

## Common mistakes

- **Reading before the update.** $y = h_{t-1} C + D x$ removes the diagonal of $M$:
  the step and the matrix stop agreeing.
- **$\Delta$ missing from the input term.** $x B^\top$ instead of $\Delta\,x B^\top$;
  the step size then controls forgetting but not writing.
- **A positive exponent.** $A = -e^{A_{\log}}$ is negative. With the sign lost the
  state grows geometrically.
- **Tiling the groups.** Group $j$ serves heads $jr$ to $jr + r - 1$. `tile` gives
  0, 1, 0, 1. Invisible with one group or with one group per head.
- **The decay sum including $s$.** $\sum_{r=s}^{t}$ multiplies every entry by an
  extra $a_s$, the diagonal included.
- **$\Delta_t$ for $\Delta_s$ in the matrix.** The scale belongs to the token whose
  input is carried.
- **No mask.** $e^{-|c_t - c_s|}$ is symmetric and looks plausible; outputs then
  depend on later inputs.
- **Clamp omitted,** applied on one side only, hard-coded, or applied to the raw
  value before the softplus. Each is invisible until a step size leaves the range:
  $\operatorname{softplus}(u) < 0.001$ needs $u < -6.9$.
- **Updating the state in place.** `h.mul_(decay)` changes the tensor the caller
  still holds; a caller that continues twice from one state gets two different answers.
- **The skip connection forgotten.** $D\,x$ is not part of $M$.

## Complexity

Step: $O(H\,P\,N)$ time and state, independent of the position. Matrix: $O(H\,T^2 N)$
to build and $O(H\,T^2 P)$ to apply. For the shipped sizes (64 heads of 64, state 128)
the state is 524,288 numbers per layer.

## In the wild

Checked on 2026-11-21 by reading and running `transformers` 5.23.0, by reading
`mlx-lm` 0.32.2, and by reading the `config.json` of
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` on the Hugging Face Hub
(`mamba_num_heads` 64, `mamba_head_dim` 64, `ssm_state_size` 128, `n_groups` 8,
`time_step_min` 0.001, no `time_step_limit` key).

- **The step.** `mamba2_selective_state_update` in
  `transformers/models/nemotron_h/modeling_nemotron_h.py`: `dt + dt_bias`, softplus,
  `dA = exp(dt * A)`, `state * dA + dt * B * x`, a batched product with `C`, plus
  `hidden_states * D`. Groups are expanded so that each serves consecutive heads. The
  public test "six cached tokens" compares your step with it.
- **The clamp differs by path and by library.** `NemotronHMamba2Mixer` sets
  `time_step_limit = (config.time_step_min, inf)` and passes it to
  `mamba2_chunk_scan`, which clamps; the one-token function receives no limit and
  does not clamp. `mlx-lm` (`models/ssm.py`, `compute_dt`) clips in both of its paths
  with the configuration's `time_step_limit`, which its Nemotron loader sets to
  `(0.0, inf)` when the key is absent, as it is in the shipped file; the default of
  `ssm_update` itself is `(0.001, 100.0)`. This problem takes the limits as an
  argument and applies them in the step, the `mlx-lm` rule, so that the step and the
  matrix form describe one layer. The hidden test "nine tokens with the clamp"
  compares a loop of your step, and your matrix, with `mamba2_chunk_scan` under
  `(0.001, inf)`, with four step sizes below the limit.
- **The matrix.** `mamba2_chunk_scan` builds exactly this within each chunk
  (`G = einsum(C, B)`, `L = exp(segment_sum(A * dt))`, `M = G * L`) and carries the
  state between chunks; `ssm_attn` in `mlx-lm` calls it the
  `surrogate_attention_matrix`.
- **Around it.** The block projects the input to a gate, to $x$, $B$, $C$ (through a
  causal convolution of kernel 4 and SiLU), and to `dt_raw`; after the scan it gates
  and then normalizes in groups, `rms_norm(y * silu(gate))`, the opposite order of
  the delta layer.
- **The paper.** "Transformers are SSMs" (arXiv 2405.21060) is where the matrix form
  comes from; only its abstract was read for this problem, the formulas here follow
  the code.

Deliberate simplifications: no batch axis; the convolution, gate, and norm of the
block are left out; `ssd_matrix` covers a whole sequence from a zero state instead of
chunks with a carried state.
