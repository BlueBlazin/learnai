## The idea

An MLX array is a node in a computation graph. Operations like `model(tokens)`, `mx.argmax`, and `+` only append nodes without running them. Calling `mx.eval(x)` computes `x` along with every unevaluated ancestor it depends on, after which those nodes store concrete values and release their inputs. An evaluated array becomes a leaf, so building new operations on top of it starts a fresh, small graph.

Think of `mx.eval` as a *cut* across the graph. Everything a step hands to the next step, or returns to the caller, should cross that cut as an evaluated array. In this loop, three values outlive a step:

| what | who reads it next | computed by `mx.eval(token)` alone? |
|---|---|---|
| the token | the next model call, the caller | yes |
| the cache | the next model call | yes, because the token comes from the logits, which depend on every cache array |
| the running log-probability | the next step's `+`, the caller | **no**, because nothing the token depends on reads it |

The third row is the main trap. An accumulator sitting off to the side of the main graph is untouched when you only evaluate `token`. Each step adds operations to it (indexing the step's log-probabilities and adding to the running sum) and keeps that step's log-probabilities alive in memory. After 200 tokens, a single "scalar" hides a chain of 600 pending operations.

Passing all three values to one `mx.eval` call evaluates their shared graph once. Including `cache` is redundant for this specific model and costs nothing because MLX skips already-evaluated arrays. It is still a good habit, because caches with auxiliary state that logits do not read (such as a rotating buffer's write position or usage statistics) suffer from the same leak as the accumulator.

## Reference solution (MLX)

```python
def generate(model, prompt, n):
    logits, cache = model(mx.array(prompt))  # prefill: a graph, nothing is computed yet
    logprob = mx.array(0.0)
    tokens = []
    for step in range(n):
        last = logits[-1]
        logprobs = last - mx.logsumexp(last)
        token = mx.argmax(logprobs)
        logprob = logprob + logprobs[token]
        mx.eval(token, logprob, cache)
        tokens.append(token)
        if step + 1 < n:  # the last token is returned, never fed back
            logits, cache = model(token.reshape(1), cache)
    return tokens, logprob
```

The prefill call needs no separate `mx.eval` because its graph is computed as part of the first step's evaluation. `token.reshape(1)` is lazy, but it adds just 1 operation on top of an evaluated leaf and joins the next step's graph. Indexing with `logprobs[token]` uses an MLX array directly, so no Python `int` conversion is ever needed.

## What the placements cost

Measured on an M1 Max (MLX 0.32.4), generating 200 tokens from a 4-token prompt (best of four runs). "Tiny" is the default `MLXTinyLM`, while "larger" has 6 layers of width 256 and a vocabulary of 8192.

| placement | tiny | larger | pending operations handed to the last model call |
|---|---|---|---|
| one `mx.eval(token, logprob, cache)` per token | 141 ms | 467 ms | 1 |
| two `.item()` reads per token | 205 ms | 508 ms | 0 |
| two `mx.eval` calls per token | 252 ms | 510 ms | 1 |
| nothing in the loop, one `mx.eval` at the end | 103 ms | 299 ms | about 243 per step: 48,000 |

Every extra evaluation pays a fixed overhead for scheduling a graph, dispatching to the GPU, and synchronizing. In this table, that overhead is 0.2 to 0.5 ms per token, which is comparable to the tiny model's entire step (0.7 ms) and nearly a tenth of the larger model's step (2.3 ms).

Deferring all evaluation until the end is faster in total because it pays that fixed overhead once. However, it produces an unusable generation loop. No token exists until the last one finishes, so you cannot stream output, check stop conditions, or inspect sampled values, and the graph (along with every intermediate array it retains) grows linearly with generation length. The MLX documentation recommends evaluating once per outer-loop iteration — neither per operation nor only at the end. In token generation, that outer iteration is the token.

## Common mistakes

- **Reading with `.item()` inside the loop.** `token.item()` evaluates the token and cache, and `logprobs[token].item()` triggers a second evaluation. That forces two implicit evaluations per step and copies the token off the GPU only to send it back.
- **Evaluating only at the end.** The numbers are right, but the graph grows without bound. The second model call receives 240 pending operations, the eighth receives 1,698, and the sixtieth receives 14,334.
- **Evaluating `token` (and `cache`) without `logprob`.** The model always receives evaluated inputs, but `logprob` returns as an unevaluated chain across all steps. The caller's first `.item()` pays for the entire chain at once, and every step's logits stay allocated until then.
- **Splitting a step's evaluation.** Calling `mx.eval(token)` followed by `mx.eval(logprob, cache)` schedules two graphs where one would suffice.
- **Evaluating the prefill separately.** Adding an `mx.eval` after prefill "to warm up" uses $n + 1$ evaluations instead of $n$. The first step's `mx.eval` already computes the prefill graph.
- **Feeding the last token back into the model.** Even if nothing evaluates that extra forward pass, the graph is still constructed. Adding `logits` to `mx.eval` immediately turns it into a wasted forward pass on every generation.
- **Summing raw logits instead of log-probabilities.** Raw logits are unnormalized and cannot be summed across steps without subtracting `mx.logsumexp(last)`.

## Complexity

$n$ evaluations, $n$ forward calls, and $P + n - 1$ tokens processed by the model. The live graph at any point covers only one step (a few hundred operations for this model), independent of $n$.

## In the wild

`mlx_lm.generate_step` uses this exact loop with one refinement covered in `mlx-streaming-generate`. It calls `mx.async_eval` on step $t + 1$ before reading the token from step $t$, keeping the GPU busy while Python processes the previous token. The cut stays in the same place, with one evaluation per token covering both the token and the log-probabilities.

When an MLX loop is unexpectedly slow, `mx.export_to_dot(file, x)` — the same tool the judge uses to count pending operations — lets you inspect the unevaluated graph behind an array and check whether something is leaking across steps.
