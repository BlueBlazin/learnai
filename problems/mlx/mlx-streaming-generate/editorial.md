## The idea

`mx.eval(x)` schedules the computation graph behind `x` and blocks until it finishes. `mx.async_eval(x)` only schedules the graph and returns immediately while the GPU works in the background. Reading `x` later (with `.item()` or any other operation that needs a concrete value) waits only for whatever work on `x` is still unfinished.

Separating scheduling from waiting lets a generation loop overlap CPU and GPU work. Python needs the integer value of token $t$ to detokenize, print, or check stop conditions, but building step $t + 1$ does not. The model accepts token $t$ as an MLX array, and an array that is still computing on the GPU can be used directly as input to the next graph. Each iteration therefore runs in three phases:

1. Build and schedule step $t + 1$ using the array for token $t$.
2. Read token $t$ with `.item()`, waiting only for step $t$ to finish.
3. Detokenize and yield token $t$ so the consumer can process it while step $t + 1$ runs on the GPU.

When the consumer asks for token $t + 1$, it is usually already computed, and the loop immediately schedules step $t + 2$. Reversing phases 1 and 2 still produces the exact same tokens, but leaves the GPU idle during phase 3 on every token.

## Reference solution (MLX)

```python
def stream_generate(model, prompt, n, detok):
    def step(tokens, cache):
        logits, cache = model(tokens, cache)
        logprobs = logits[-1] - mx.logsumexp(logits[-1])
        token = mx.argmax(logprobs)
        return token, logprobs[token], cache

    token, logprob, cache = step(mx.array(prompt), None)
    mx.async_eval(token, logprob)
    for index in range(n):
        if index + 1 < n:
            next_token, next_logprob, cache = step(token.reshape(1), cache)
            mx.async_eval(next_token, next_logprob)
        yield detok(token.item()), logprob
        if index + 1 < n:
            token, logprob = next_token, next_logprob
```

Three details make this loop work cleanly.

**What to pass to `async_eval`.** Include every output the consumer or the next step will use that `token` does not already depend on. The KV cache is an ancestor of `token` because computing `logits` reads every key and value, so scheduling `token` schedules the cache automatically. By contrast, `logprob` (`logprobs[token]`) is a *descendant* of `token`. If you omit `logprob` from `mx.async_eval`, it is yielded as an unevaluated two-operation graph that forces a synchronous evaluation when the consumer reads it, just like the accumulator in `mlx-lazy-eval-placement`.

**Stopping at the last token.** Because the loop knows `n` in advance, it checks `index + 1 < n` and skips scheduling step $n$. Stop tokens behave differently in real systems because you cannot tell whether token $t$ is a stop token until you read it, by which time step $t + 1$ is already in flight. Paying one extra forward pass for a stop token is unavoidable with overlap, but paying it at a known length limit is not.

**Generator semantics.** Nothing runs until the consumer requests the first item, and execution pauses at every `yield`. Any work the consumer performs between iterations runs directly in parallel with the GPU step already in flight.

## What the overlap is worth

Measured on an M1 Max with MLX 0.32.4, generating 100 tokens from a 4-token prompt (best of four runs, milliseconds per token). "Larger" is an `MLXTinyLM` with 6 layers of width 256 and vocabulary size 8192. "Wide" has 8 layers of width 1024. Per-token Python work is simulated by a `detok` function that spins for the listed duration.

| model | Python work per token | `mx.eval`, then read | `mx.async_eval` of the next step, then read |
|---|---|---|---|
| tiny | none | 0.73 | 0.49 |
| larger | none | 2.30 | 1.30 |
| larger | 0.5 ms | 2.76 | 1.72 |
| wide | none | 3.15 | 2.21 |
| wide | 0.5 ms | 4.36 | 2.25 |
| wide | 2.0 ms | 5.82 | 3.74 |

Two effects show up in the numbers. Even with zero Python work, the async loop is faster because building the graph for step $t + 1$ (a few hundred microseconds in Python) happens while the GPU computes step $t$. In addition, Python work up to the duration of a GPU step is hidden for free. The wide model takes about 2.2 ms per token whether detokenization takes 0 ms or 0.5 ms, whereas the blocking loop pays the full 0.5 ms on top. Once Python work exceeds a GPU step (the last row), Python becomes the bottleneck and the GPU step hides inside it.

## Common mistakes

- **Using `mx.eval` before reading.** Adding `yield` to the blocking loop from `mlx-lazy-eval-placement` produces the right tokens serially and hits the slower left column of the benchmark table.
- **Scheduling after reading.** Calling `mx.async_eval` only after reading and detokenizing token $t$ eliminates all overlap, turning asynchronous evaluation into blocking evaluation with extra overhead. Converting the token to a Python `int` before feeding it back (`mx.array([token.item()])`) causes the same bug by forcing a read before step $t + 1$ can be built.
- **Scheduling a step after the last token.** Unconditionally scheduling the next step launches a forward pass whose token is never yielded. Because `mx.async_eval` submits work to the GPU immediately, that wasted step actually executes.
- **Scheduling only the token and omitting `logprob`.** The token computes in the background while `logprob` is yielded lazy. Whoever reads `logprob` pays a synchronous evaluation, and until then the unevaluated graph keeps that step's logits alive.
- **Collecting all pairs before yielding.** Running the entire loop into a list and yielding at the end produces the right tokens and evaluation order, but makes time to first token equal to time to last token.
- **Advancing before yielding.** Overwriting `token` with `next_token` before `yield` drops the first token generated from the prompt, shifting the entire output by one position.

## Complexity

The loop performs $n$ scheduled evaluations, $n$ reads, $n$ forward passes, and processes $P + n - 1$ tokens in total. At most two steps exist in flight at any moment: the step being read and the next step computing on the GPU.

## In the wild

`mlx_lm.generate.generate_step` uses this exact loop structure with a sampler and logits processors in place of argmax, calling `mx.async_eval(next_y, next_logprobs)` before yielding `y.item()`. Its cache objects are mutated in place by the model, so the cache state is still an ancestor of the token. Unlike this problem, `mlx_lm` treats a token limit the same way as a stop token, so one extra forward pass is already scheduled when the limit is hit. Although `mx.async_eval` is marked experimental in the MLX documentation, this two-step prefetching pattern is standard across MLX inference loops. Because `EvalSpy` (the judge utility that records evaluation order) cannot intercept implicit evaluations triggered by `if x:` or f-strings, the test harness also verifies overlap by checking `model.counters.forward_calls` inside `detok` on every token.
