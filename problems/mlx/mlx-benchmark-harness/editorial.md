## The idea

A benchmark measures where time is actually spent. In an eager framework, reading the clock before and after a model call brackets the computation because the work runs immediately. MLX breaks that assumption. Calling the model only builds a lazy graph and returns right away, while the real work executes later inside whichever call first forces evaluation. Clock readings therefore bracket *evaluation*, not the lines that construct the graph.

This leads to one rule applied in three places. **Before every clock reading that ends a timed interval, evaluate everything that belongs to that interval.**

1. **Before $t_1$.** Call `mx.eval(token)` right after the prefill. Without it, the prompt interval only measures graph construction, and the prefill compute gets deferred into the first decode step and billed to generation.
2. **At every decode step (and thus before $t_2$).** Call `mx.eval(token)` on each iteration of the decode loop. Evaluating once at the end of the loop would produce the same total on the simulated clock, but it times a different workload than real generation. A real decode loop must materialize each token to emit it, whereas a deferred loop skips per-step synchronization and grows a larger graph.
3. **Inside the warm-up runs.** Warm-up pays one-time setup costs before measurement begins. Metal compiles kernels the first time an operation runs with a given signature, and the memory allocator populates its buffer cache. Both happen during evaluation, so a lazy warm-up warms nothing and leaves the cold-start cost in the first measured run.

`mx.eval` blocks until its arguments are computed, so no separate synchronization call is needed. `mx.synchronize()` is only required when work was launched asynchronously with `mx.async_eval`. On an M1 Max, calling `mx.synchronize()` after `mx.eval` on a 20 ms matrix multiplication returns in 0.08 ms. Calling `mx.async_eval` on the same operation returns in 0.05 ms, and the following `mx.synchronize()` takes the full 20 ms.

Individual runs also suffer from occasional slowdowns caused by background processes, thermal throttling, or fresh memory allocations. Because this noise is one-sided, the **median** is much more stable than the mean. A single slow outlier out of five runs shifts the mean by a fifth of its excess latency while leaving the median unchanged. Finally, each throughput rate divides the tokens processed in *its* phase by the time of *its* phase: $P$ divided by the median prompt time, and `n_tokens` divided by the median generation time.

## Reference solution (MLX)

```python
def benchmark(model, prompt, n_tokens, n_warmup, n_runs, clock):
    prompt_tokens = mx.array(prompt)

    def run():
        start = clock()
        logits, cache = model(prompt_tokens)
        token = mx.argmax(logits[-1])
        mx.eval(token)
        first_token = clock()
        tokens = [token]
        for _ in range(n_tokens):
            logits, cache = model(token.reshape(1), cache)
            token = mx.argmax(logits[-1])
            mx.eval(token)
            tokens.append(token)
        end = clock()
        return first_token - start, end - first_token, tokens

    for _ in range(n_warmup):
        run()
    runs = [run() for _ in range(n_runs)]
    prompt_time = median(times for times, _, _ in runs)
    generation_time = median(times for _, times, _ in runs)
    return {
        "prompt_tps": len(prompt) / prompt_time,
        "generation_tps": n_tokens / generation_time,
        "tokens": [token.item() for token in runs[-1][2]],
    }
```

Evaluating `token` materializes the entire phase because `token` depends on `logits`, which in turn depends on every array written into `cache`. Nothing built during the phase remains pending. Keeping tokens as MLX arrays during the loop and converting them to Python ints after the final clock reading avoids timing Python conversion overhead. Reusing the same `run` helper for both warm-up and measured runs guarantees that warm-up executes the exact same code path.

## What the mistakes report

Using the worked example (true rates: 493.8 and 243.9 tokens per second), here is what each common mistake reports on the simulated clock:

| harness | prompt tokens/s | generation tokens/s |
|---|---|---|
| correct | 493.8 | 243.9 |
| no evaluation inside the timed regions | 40,000 | 10,000 |
| no evaluation before the middle reading | 40,000 | 196.1 |
| warm-up runs built but not evaluated | 330.6 | 163.9 |
| warm-up run included in the statistics | 396.0 | 196.1 |
| mean in place of median | 424.0 | 209.8 |
| generation time measured from the start of the run | 493.8 | 195.6 |
| `n_tokens + 1` tokens over the generation time | 493.8 | 274.4 |

Here is the same comparison on real hardware (M1 Max, MLX 0.32.4) using `time.perf_counter`, an 8-layer `MLXTinyLM` of width 1024, a 256-token prompt, and 32 decode steps:

| what was timed | prompt | generation |
|---|---|---|
| evaluated, first run of the process | 35.1 ms (7,300 tokens/s) | 107.7 ms (297 tokens/s) |
| evaluated, later runs | 17.6 to 27.5 ms, median 20.9 ms | 105 to 118 ms |
| not evaluated | 0.4 ms (about 600,000 tokens/s) | 10 to 14 ms (about 2,800 tokens/s) |

The unevaluated prompt time measures only prefill graph construction. The unevaluated generation time measures 32 decode graph constructions at 0.3 to 0.4 ms each, which is the per-token Python overhead that `mx.async_eval` hides in `mlx-streaming-generate`. Notice that the cold run's prompt time is roughly twice the warm median, and even warm prompt times range from 17.6 to 27.5 ms on an idle machine, which is why the median is necessary.

## Common mistakes

- **Timing a lazy graph.** Skipping `mx.eval` inside the timed region and evaluating only afterward to extract the tokens measures graph construction alone. The reported rates are inflated by two orders of magnitude.
- **Not evaluating before the middle reading.** If the decode loop evaluates per token, the total runtime is right, but the prefill executes during the first decode step. Prompt processing looks instant while generation is penalized by the full prefill time.
- **Running a lazy warm-up.** Writing `for _ in range(n_warmup): model(prompt_tokens)` builds and discards graphs without executing them. Nothing is compiled or allocated, so the first measured run pays the cold-start cost.
- **Keeping warm-up runs in the statistics.** Computing the median over all `n_warmup + n_runs` timings folds the cold run back into the result.
- **Using the mean instead of the median.** One slow run out of three shifts the mean in the worked example by 14 percent. Because latency noise is one-sided, report the median (and percentiles rather than standard deviation if you report spread).
- **Taking the upper median.** `sorted(times)[n // 2]` works for odd counts but picks the upper middle value for even counts. Use `statistics.median`, which averages the two middle values.
- **Measuring generation time from the start of the run.** Using `end - start` measures end-to-end time, making the generation rate depend on prompt length. This is the same error as folding time to first token into time per output token in `latency-metrics`.
- **Counting the first token in generation.** A run outputs `n_tokens + 1` tokens, but the first token was produced during the prompt phase. Dividing `n_tokens + 1` by the generation time inflates the rate, doubling it when `n_tokens == 1`.
- **Evaluating only once per phase.** Deferring all decode steps until a single `mx.eval` before $t_2$ times a loop nobody runs in production and passes an unevaluated graph into every step.

## Complexity

The harness performs $(\text{n\_warmup} + \text{n\_runs}) \cdot (1 + \text{n\_tokens})$ model calls and evaluations, takes three clock readings per run, and stores one run's token list.

## In the wild

`mlx_lm.generate.stream_generate` times generation with `time.perf_counter` in the same two phases. The prompt interval ends when the first token arrives (after the underlying generator has evaluated the prefill), and `prompt_tps` and `generation_tps` come from those two intervals. `mlx_lm.benchmark` also discards one warm-up generation before its timed trials.

Two choices in `mlx_lm.benchmark` differ from this problem and are worth noting when reading its numbers. First, its generation rate divides *all* generated tokens (including the first) by the time after the first token arrives, multiplying the rate by $(\text{n\_tokens} + 1) / \text{n\_tokens}$. That difference is tiny for hundreds of tokens, but it doubles the rate when $\text{n\_tokens} = 1$. Second, it reports the mean of per-trial rates rather than a median. Whenever you compare throughput numbers across tools, check what was evaluated before each clock reading, whether the first run was discarded, and which tokens were divided by which interval.
