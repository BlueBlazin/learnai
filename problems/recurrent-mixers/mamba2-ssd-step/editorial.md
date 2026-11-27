## The idea

Each channel of each head runs a linear recurrence with a scalar decay $a_t = e^{-\Delta_t e^{A_{\log}}}$:

$$
h_t = a_t\,h_{t-1} + \Delta_t\,x_t\,B_t^\top, \qquad y_t = h_t\,C_t + D\,x_t .
$$

This is a discretized differential equation $\dot h = A h + x B^\top$ with $A = -e^{A_{\log}} < 0$, integrated over a step $\Delta_t$ chosen by the token. A large step forgets much of the prior state and writes strongly, while a small step preserves the state and writes weakly. Softplus keeps $\Delta_t$ positive, and clamping keeps it inside the configured bounds.

Unrolling the recurrence from $h_{-1} = 0$ gives

$$
h_t = \sum_{s \le t} \Bigl(\prod_{r = s+1}^{t} a_r\Bigr)\,\Delta_s\,x_s\,B_s^\top
\quad\Longrightarrow\quad
y_t = \sum_{s \le t} \underbrace{(C_t \cdot B_s)\; e^{-e^{A_{\log}} \sum_{r=s+1}^{t} \Delta_r}\; \Delta_s}_{M_{t,s}}\; x_s + D\,x_t .
$$

This matches the structure of attention. The score between query $t$ and key $s$ is the dot product $C_t \cdot B_s$, the mask is a causal decay determined by the step sizes between positions $s$ and $t$, and the values are the inputs scaled by their own step sizes $\Delta_s$. There is no softmax and no learned positional bias. The decay product starts at $r = s + 1$ because token $s$ decays the state before writing, so its own step does not decay its own input.

The two forms trade parallelism for state size. The matrix form costs $O(T^2)$ and runs as a batched matrix product, making it fast for chunked prefill when state is carried between chunks. The step form costs $O(1)$ per token and stores only $h$, making it ideal for decoding. A serving engine needs both, and they must agree numerically.

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

- **Reading before updating the state.** Computing $y = h_{t-1} C + D x$ removes the diagonal of $M$, breaking agreement between `ssd_step` and `ssd_matrix`.
- **Omitting $\Delta$ from the input term.** Writing $x B^\top$ instead of $\Delta\,x B^\top$ lets the step size control forgetting but not writing.
- **Dropping the minus sign in the exponent.** Because $A = -e^{A_{\log}}$ is negative, losing the minus sign makes the state grow exponentially.
- **Tiling groups instead of repeating them.** With $r = H / G$ heads per group, group $j$ serves consecutive heads $jr$ through $jr + r - 1$. Using `tile` produces the pattern `0, 1, 0, 1`, which passes unnoticed when $G = 1$ or $G = H$.
- **Including $s$ in the decay sum.** Summing from $r = s$ to $t$ multiplies every entry, including the diagonal, by an extra factor of $a_s$.
- **Scaling by $\Delta_t$ instead of $\Delta_s$ in `ssd_matrix`.** The scale factor belongs to the source token $s$ whose input is being mixed.
- **Omitting the causal mask.** Using a symmetric decay like $e^{-|c_t - c_s|}$ lets outputs depend on future tokens.
- **Mishandling the clamp.** Omitting the clamp, clamping on only one side, hard-coding bounds, or clamping before softplus all fail once step sizes leave the range (note that $\operatorname{softplus}(u) < 0.001$ requires $u < -6.9$).
- **Updating `h` in place.** Calling `h.mul_(decay)` mutates the caller's state tensor, so branching twice from the same state yields corrupted results.
- **Forgetting the skip connection.** The term $D\,x$ is separate from $M$.

## Complexity

`ssd_step` takes $O(H\,P\,N)$ time and state, independent of sequence length. `ssd_matrix` takes $O(H\,T^2 N)$ time to construct and $O(H\,T^2 P)$ time to apply. At the shipped Nemotron dimensions (64 heads of width 64 with state size 128), the recurrent state holds 524,288 numbers per layer.

## In the wild

Checked on 2026-11-21 by reading and running `transformers` 5.23.0, reading `mlx-lm` 0.32.2, and inspecting `config.json` for `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` on the Hugging Face Hub (`mamba_num_heads` 64, `mamba_head_dim` 64, `ssm_state_size` 128, `n_groups` 8, `time_step_min` 0.001, and no `time_step_limit` key).

- **The step.** `mamba2_selective_state_update` in `transformers/models/nemotron_h/modeling_nemotron_h.py` computes `dt + dt_bias`, applies softplus, evaluates `dA = exp(dt * A)`, updates `state * dA + dt * B * x`, contracts with `C`, and adds `hidden_states * D`. Groups are expanded so each group serves consecutive heads. The public test "six cached tokens" compares your `ssd_step` against this function.
- **Clamp differences across paths and libraries.** `NemotronHMamba2Mixer` sets `time_step_limit = (config.time_step_min, inf)` and passes it to `mamba2_chunk_scan`, which clamps. However, the single-token function receives no limit and does not clamp. In `mlx-lm` (`models/ssm.py`, `compute_dt`), both paths clip using `time_step_limit`, which the Nemotron loader sets to `(0.0, inf)` when the config key is absent (as in the shipped file), while `ssm_update` itself defaults to `(0.001, 100.0)`. This problem passes `dt_limit` explicitly and applies it in `ssd_step` (following `mlx-lm`) so the step and matrix forms describe one consistent layer. The hidden test "nine tokens with the clamp" compares both a loop of your step and your matrix against `mamba2_chunk_scan` under `(0.001, inf)` with four step sizes below the limit.
- **The matrix.** `mamba2_chunk_scan` builds this exact matrix within each chunk (`G = einsum(C, B)`, `L = exp(segment_sum(A * dt))`, `M = G * L`) and carries the state across chunks. In `mlx-lm`, `ssm_attn` calls it the `surrogate_attention_matrix`.
- **Surrounding block.** The full mixer projects the input into a gate, $x$, $B$, $C$ (passed through a causal convolution of kernel size 4 and SiLU), and `dt_raw`. After the scan, it gates and then normalizes in groups via `rms_norm(y * silu(gate))`, which is the reverse of the gated delta layer's order.
- **The paper.** The matrix formulation comes from "Transformers are SSMs" (arXiv 2405.21060). Only the abstract was consulted for this problem, and all formulas here follow the model code directly.

Deliberate simplifications: there is no batch axis, the surrounding convolution, gate, and norm are omitted, and `ssd_matrix` covers a full sequence from a zero state rather than chunks with a carried state.
