Implement a token-sampling decode loop that runs on the GPU without synchronizing with the host on every token. The sampling step, output buffer, and end-of-sequence tracking must all stay on `device`, so the host reads from the device only once every `check_every` tokens and once at the end.

A naive loop that calls `.item()` and checks `if token == eos_id` produces the right tokens, but it forces the CPU to wait for the GPU after every single token.

## Specification

```python
def decode(model, prompt, n, uniforms, device, eos_id, check_every): ...
```

`model` is a small causal transformer with a KV cache, already placed on `device` (the `StepLM` class in `tests.py`). You can call it in two ways:

```python
logits, cache = model(tokens)          # no cache: `tokens` is the full sequence so far
logits, cache = model(tokens, cache)   # with cache: `tokens` extends the cached prefix
```

- `tokens` must be a 1-D int64 tensor on `device` with length $T \ge 1$. Python lists and tensors on other devices are rejected.
- `logits` has shape `(T, V)` on `device`, where $V$ is the vocabulary size. Row `logits[-1]` contains the unnormalized scores for the next token.
- `cache` is opaque. Each call returns a new cache without modifying the previous one.
- `model.counters.forward_calls` counts calls to `model`, and `model.counters.tokens_processed` counts the total number of tokens passed in.

The remaining arguments are:

- `prompt`: initial token IDs of length $P \ge 1$, provided on the CPU.
- `n`: maximum number of tokens to generate ($n \ge 1$).
- `uniforms`: random draws in $[0, 1)$, one per step, provided on the CPU. Generated token $i$ (0-indexed) uses `uniforms[i]`.
- `device`: the `torch.device` where `model` lives.
- `eos_id`: the end-of-sequence (EOS) token ID.
- `check_every`: an integer $\ge 1$ setting how often the host may read from `device`.

**Sampling.** Let $\ell \in \mathbb{R}^V$ be the next-token logits `logits[-1]`, let $p = \operatorname{softmax}(\ell)$ be the token probabilities, and let $c_k = \sum_{j=0}^{k} p_j$ be their cumulative sum. Given the step's uniform draw $u = \texttt{uniforms}[i]$, select the first token index whose cumulative probability exceeds $u$ times the total probability (inverse-CDF sampling, as in `categorical-sample`):

$$
\text{token} = \min \{\, k : c_k > u \cdot c_{V-1} \,\}.
$$

**End of sequence.** Generation ends once `eos_id` is sampled, and that first `eos_id` is included in the output. However, the host may check whether EOS has occurred only after generated tokens `check_every`, `2 * check_every`, and so on (1-indexed). If an EOS was sampled by a checkpoint, the loop stops there. Between an EOS and the next checkpoint, the loop continues for up to `check_every - 1` extra steps, and any tokens sampled during those extra steps must not appear in the result.

**Result.** Return an int64 tensor of shape `(n,)` on the CPU. It contains the sampled tokens up to and including the first `eos_id`, followed by `eos_id` in every remaining position. If EOS is never sampled, it contains all `n` sampled tokens. The output shape is always `(n,)`, regardless of when EOS occurs. Transfer the final tensor to the host with `.cpu()`.

**Limits.** The judge enforces three exact limits on your code:

1. **Host synchronizations:** at most $\lceil n / \texttt{check\_every} \rceil + 1$. Every operation that reads a tensor value onto the host is counted, including `.item()`, `.tolist()`, `int(x)`, `float(x)`, `bool(x)` (such as `if x:`, `while x:`, `x and y`), printing, and f-strings. Transfers to the host (`.cpu()`, `.numpy()`, `np.asarray(x)`) and operations with data-dependent output shapes (`nonzero`, boolean-mask indexing `x[mask]`, single-argument `torch.where(cond)`, `unique`, `masked_select`) also count. Reading a scalar flag once per checkpoint plus one `.cpu()` at the end fits the budget, while reading once per token does not.
2. **Device allocation:** every tensor factory call (`torch.zeros`, `torch.full`, `torch.tensor`, `torch.arange`, and similar) must pass `device=`. Omitting `device=` fails the test.
3. **Model work:** let $s$ be the stopping step. If EOS is never sampled, $s = n$. Otherwise, $s$ is the smallest multiple of `check_every` greater than or equal to the 1-based position of the first EOS, capped at $n$. Your code must satisfy `forward_calls` $\le s$ (one call for the prompt, then one call per generated token before stopping) and `tokens_processed` $\le P + s - 1$.

## Shapes

| name | shape | dtype | device |
|---|---|---|---|
| `prompt` | `(P,)` | int64 | CPU |
| `uniforms` | `(n,)` | float32 | CPU |
| `logits` (from `model`) | `(T, V)` | float32 | `device` |
| result | `(n,)` | int64 | CPU |

## Example

```python
model = StepLM(seed=3)  # V = 32, eos_id = 1
prompt = [0, 7, 19, 4]
uniforms = [0.3735, 0.8579, 0.0272, ...]  # 8 numbers

decode(model, prompt, 8, uniforms, torch.device("cpu"), eos_id=1, check_every=4)
# [9, 27, 1, 1, 1, 1, 1, 1]
```

| token | fed to the model | sampled | written | host reads |
|---|---|---|---|---|
| 1 | `[0, 7, 19, 4]` | 9 | 9 | |
| 2 | `[9]` | 27 | 27 | |
| 3 | `[27]` | 1 (EOS) | 1 | |
| 4 | `[1]` | some token | 1 | reads flag: EOS seen, stop |

Token 3 is `eos_id`, but the first checkpoint is after token 4. The loop therefore runs one extra step and samples a fourth token, which is masked out so position 4 still receives `eos_id`. Positions 5 through 8 are never computed and remain `eos_id`.

In total, the loop makes 4 model calls, processes 7 tokens, and performs 2 host synchronizations (one flag read and one `.cpu()`), which is within the limit of $\lceil 8 / 4 \rceil + 1 = 3$.

## Constraints

- $1 \le P \le 8$, $1 \le n \le 64$, and $1 \le \texttt{check\_every} \le 16$. The vocabulary has $V = 32$ tokens.
- Output tokens are compared exactly. Every test chooses `uniforms[i]` near the middle of its target token's interval, at least `2e-4` away from every cumulative sum, so float32 rounding in softmax or `cumsum` cannot change the sampled token.
- All three limits are exact counts. Exceeding any limit fails the test even if the output tokens match.
- Moving inputs to the device with `.to(device)` is not counted, and you should move `prompt` and `uniforms` once before the loop. Do not use `.to("cpu")` to return the result; use `.cpu()` once at the end.
- **What runs where.** Standard test cases run on the CPU, where host reads are cheap and the judge counts them directly. Two hardware-marked cases run only on an NVIDIA GPU with the model on CUDA and enforce the same checks. The GPU cases were written from the PyTorch 2.15 documentation and could not be run when the problem was authored.

## Why this matters

A GPU executes asynchronously ahead of Python. Each PyTorch op queues a kernel and returns immediately so the CPU can keep queuing work. Reading a single scalar back to Python (via `.item()` or a tensor in an `if` condition) breaks that pipeline. The CPU stalls until every queued kernel finishes and the value is copied over.

In a decode loop, doing this on every token forces the GPU to idle while Python processes the token and forces Python to idle while the GPU computes the next step. Checking for EOS only every `check_every` tokens trades up to `check_every - 1` wasted steps and a fixed-shape padded output for uninterrupted GPU execution. Production serving engines make this exact tradeoff, and compiled or CUDA-graph-captured decode steps (covered next in this track) cannot contain host reads at all.
