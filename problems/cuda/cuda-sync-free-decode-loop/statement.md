Sample tokens from a language model in a loop that stays on the GPU: the sampling, the
token buffer, and the end-of-sequence test all live on the device, and the host looks
at a value only once every `check_every` tokens and once at the end. The obvious loop,
with `.item()` and `if token == eos_id`, is correct and makes the host wait for the
device at every token.

## Specification

```python
def decode(model, prompt, n, uniforms, device, eos_id, check_every): ...
```

`model` is a small causal transformer with a KV cache, written in torch and already
on `device` (the class `StepLM` in this problem's `tests.py`). It is called in two
ways:

```python
logits, cache = model(tokens)  # no cache: `tokens` is the whole sequence so far
logits, cache = model(tokens, cache)  # with a cache: `tokens` continue the cached prefix
```

- `tokens` must be a 1-D int64 **tensor on `device`** with at least one token. A
  Python list or a tensor on another device is refused.
- `logits` is a float32 tensor of shape `(T, V)` on `device`, one row per token of
  this call; `logits[-1]` scores the next token. `cache` is opaque; the model never
  modifies one and returns a new one from every call.
- `model.counters.forward_calls` counts the calls and
  `model.counters.tokens_processed` the tokens passed in.

The other arguments:

- `prompt`: an int64 tensor of shape `(P,)`, $P \ge 1$, on the CPU.
- `n`: the maximum number of tokens to generate, $n \ge 1$.
- `uniforms`: a float32 tensor of shape `(n,)` on the CPU, each in $[0, 1)$.
  Randomness is an input: generated token $i$ (counting from 0) is sampled with
  `uniforms[i]`.
- `device`: the `torch.device` of the model.
- `eos_id`: the end-of-sequence token id.
- `check_every`: an int $\ge 1$, how often the host may look at the device.

**Sampling.** With $\ell$ the next-token logits, $p = \operatorname{softmax}(\ell)$,
$c_k = \sum_{j \le k} p_j$, and $u$ the uniform number of this step, the token is

$$
\text{token} = \min \{\, k : c_k > u \cdot c_{V-1} \,\},
$$

the first index whose cumulative probability exceeds $u$ times the total (inverse-CDF
sampling, as in `categorical-sample`).

**End of sequence.** Generation ends with the first sampled `eos_id`, which is part
of the result. But the host learns about it late: the loop may read from the device
only after generated tokens number `check_every`, `2 * check_every`, ... (counting
from 1). At such a point, if an EOS has been sampled, the loop stops. So after an
EOS the loop runs on until the next check, at most `check_every - 1` more tokens,
and whatever those steps sample must not appear in the result.

**Result.** An int64 tensor of shape `(n,)` **on the CPU**: the sampled tokens up to
and including the first `eos_id`, and `eos_id` in every position after it. Without
an EOS it holds all `n` sampled tokens. The shape never depends on where the EOS
fell. Bring the result to the host with `.cpu()`.

**Limits.** The judge counts what the code in your file does:

1. **Host synchronizations: at most $\lceil n / \texttt{check\_every} \rceil + 1$.**
   Counted are every read of a tensor's value (`.item()`, `.tolist()`, `int(x)`,
   `float(x)`, `bool(x)` and so `if x:`, `while x:`, `x and y`, printing, an
   f-string), every transfer to the host (`.cpu()`, `.numpy()`, `np.asarray(x)`), and
   every operation whose result has a data-dependent size (`nonzero`, `x[mask]` with
   a boolean mask, `torch.where(cond)` with one argument, `unique`,
   `masked_select`). One read of the flag per check and one `.cpu()` at the end fit;
   one read per token does not.
2. **Tensors are created on the device.** A factory call (`torch.zeros`,
   `torch.full`, `torch.tensor`, `torch.arange`, ...) without `device=` fails the
   case.
3. **Model work.** Let $s$ be the number of tokens after which the loop stops:
   $s = n$ without an EOS, and otherwise the smallest multiple of `check_every` that
   is at least the position of the EOS (counting from 1), capped at $n$. Then
   `forward_calls` $\le s$ (the prompt in one call, then one call per generated token
   except the last) and `tokens_processed` $\le P + s - 1$.

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
| 4 | `[1]` | some token | 1 | the flag: an EOS was sampled, stop |

Token 3 is the EOS, but the first check is after token 4, so the model is called
once more and step 4 samples a token that nobody wants; position 4 of the result is
`eos_id` all the same, and so are positions 5 to 8, which were never computed. In
total: 4 model calls, 7 tokens processed, and 2 host synchronizations (one read of
the flag, one `.cpu()`), within the limit of $\lceil 8 / 4 \rceil + 1 = 3$.

## Constraints

- $1 \le P \le 8$, $1 \le n \le 64$, $1 \le \texttt{check\_every} \le 16$; the
  vocabulary has 32 tokens.
- The tokens are compared exactly. The tests choose every `uniforms[i]` in the middle
  of its token's interval, at least `2e-4` away from every cumulative sum, so
  float32 rounding in your softmax or cumulative sum cannot change a sample.
- The limits are exact counts. A solution that exceeds one fails even when its
  tokens are right.
- Moving a tensor with `.to(device)` is not counted, and is how `prompt` and
  `uniforms` reach the device (once, before the loop). Do not use `.to("cpu")` to
  bring results back: use `.cpu()`, once.
- **What runs where.** The cases without a hardware mark run on the CPU, where a
  host read costs nothing and the judge counts them instead. Two cases run only on a
  machine with an NVIDIA GPU, with the model on the CUDA device, and apply the same
  checks there. The GPU cases were written from the PyTorch 2.15 documentation and
  could not be run when the problem was authored.

## Why this matters

A GPU runs ahead of Python: each operation is queued and the host continues. Reading
one value back (`.item()`, a tensor in an `if`) forces the host to wait until
everything queued has finished and then to copy the value across. In a decode loop
that happens once per token, per sequence: the device idles while Python handles the
token, Python idles while the device computes, and the launch-ahead that makes small
models fast is gone. The fix has a cost, which this problem makes explicit: checking
for the end of the sequence only every few tokens means computing up to
`check_every - 1` tokens that are thrown away, and a result of fixed shape padded
with EOS. Serving stacks choose that trade deliberately, for instance by testing
their stopping criteria every few steps, and compiled or graph-captured decode steps
(the next problems of this track) cannot contain a host read at all.
