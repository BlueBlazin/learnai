## The idea

A benchmark is a claim about where time went. In an eager framework the claim is easy
to make: the work happens on the line that asks for it, so a clock reading before
and after the line brackets the work. MLX breaks that link. A line that calls the
model builds a graph and returns; the work happens later, inside whichever call first
needs a value, and takes its time there. A clock reading brackets *evaluation*, not
the code that described the computation.

So a harness for a lazy framework has one rule, applied three times: **before every
clock reading that ends an interval, evaluate everything the interval is supposed to
contain.**

1. *Before $t_1$*: `mx.eval(token)` after the prefill. Otherwise the prompt time is
   the time to build one graph, and the prefill is computed by the first decode step
   and billed to generation.
2. *Before $t_2$*, and at every step before it: `mx.eval(token)` per decode step.
   One evaluation at the end of the phase would give the same total on the simulated
   clock, and a different program on a real one: the loop people run evaluates per
   token (it has to, to emit the token), and a deferred loop skips that per-step cost
   while growing a graph. The harness must time the loop it claims to time.
3. *In the warm-up runs.* A warm-up exists to pay one-time costs before the
   measurement: Metal kernels are compiled the first time an operation runs with a
   given signature, and the allocator has no buffers to reuse yet. Both happen at
   evaluation. A warm-up that only builds graphs warms nothing, and the cold run is
   then the first measured one.

`mx.eval` is enough: it returns when its arguments are computed. `mx.synchronize()`
is the call that waits for work scheduled *without* waiting, which is what
`mx.async_eval` does; a harness that used `async_eval` would need it before every
clock reading. On this machine, after `mx.eval` of a 20 ms matrix product, a
following `mx.synchronize()` returns in 0.08 ms; after `mx.async_eval` of the same
product, which returns in 0.05 ms, it takes the full 20 ms.

The rest is statistics and bookkeeping. Runs vary (another process, a thermal
change, an allocation that did not come from the cache), always in the slow
direction, so the times are skewed and the **median** is the robust summary: one bad
run out of five moves the mean by a fifth of its excess and the median not at all.
And each rate divides the tokens of *its* phase by the time of *its* phase: $P$ over
the prompt time, `n_tokens` decode steps over the generation time.

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

Evaluating the token evaluates the phase: the token is computed from the logits, the
logits from every array of the cache, so nothing the phase built is left pending.
The tokens stay arrays inside the run and become ints after the last clock reading,
so no conversion is timed. The same `run` serves warm-up and measurement, which is
the simplest way to make sure a warm-up run really is a run.

## What the mistakes report

The worked example again (true rates: 493.8 and 243.9 tokens per second), with each
mistake's numbers on the simulated clock:

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

And the real thing, measured on this machine (M1 Max, MLX 0.32.4) with
`time.perf_counter`, an `MLXTinyLM` of 8 layers and width 1024, a 256-token prompt,
and 32 decode steps:

| what was timed | prompt | generation |
|---|---|---|
| evaluated, first run of the process | 35.1 ms (7,300 tokens/s) | 107.7 ms (297 tokens/s) |
| evaluated, later runs | 17.6 to 27.5 ms, median 20.9 ms | 105 to 118 ms |
| not evaluated | 0.4 ms (about 600,000 tokens/s) | 10 to 14 ms (about 2,800 tokens/s) |

The unevaluated "prompt time" is the cost of building the prefill graph. The
unevaluated "generation time" is 32 graph constructions, 0.3 to 0.4 ms each, which is
also a measurement worth knowing: it is the Python overhead per token that
`mx.async_eval` hides in `mlx-streaming-generate`. The cold run's prompt time is twice
the warm one. The spread of the warm prompt times, 17.6 to 27.5 ms on an idle
machine, is the reason for the median.

## Common mistakes

- **Timing a lazy graph.** No evaluation between the clock readings, and an
  evaluation afterwards "to get the tokens". The intervals contain graph construction
  only. The numbers are too good by two orders of magnitude, which at least looks
  suspicious; the next mistake does not.
- **Not evaluating before the middle reading.** The decode loop evaluates per token,
  so the total is right, but the prefill is computed inside the first decode step.
  Time to first token is reported as instant, and generation as slower than it is by
  exactly the prefill.
- **A lazy warm-up.** `for _ in range(n_warmup): model(prompt_tokens)` and variations
  of it build and discard graphs. Nothing is compiled or allocated, and the first
  measured run is the cold one.
- **Keeping the warm-up runs.** Performing them and then computing statistics over
  all `n_warmup + n_runs` times puts the cold run back into the result.
- **The mean.** One slow run in three moves the mean of the worked example by 14
  percent. Report the median, and if you report a spread, percentiles, not a standard
  deviation: the distribution is one-sided.
- **The upper median.** `sorted(times)[n // 2]` is the median for an odd count and
  the upper of the two middle values for an even one. `statistics.median` averages
  them.
- **Generation time from the start of the run.** `end - start` is end-to-end time.
  Dividing `n_tokens` by it gives a generation rate that depends on the length of the
  prompt, the same confusion as folding the first token into time per output token in
  `latency-metrics`.
- **Counting the first token.** A run produces `n_tokens + 1` tokens, but the first
  was computed during the prompt time. Dividing all of them by the generation time
  overstates the rate, by a factor of 2 for a single decode step.
- **One evaluation per phase.** Deferring the decode steps and evaluating once before
  $t_2$ measures a loop nobody runs, and hands every step the unevaluated graph of
  the steps before it.

## Complexity

$(\text{n\_warmup} + \text{n\_runs}) \cdot (1 + \text{n\_tokens})$ model calls and as
many evaluations, three clock readings per run, and one run's tokens kept.

## In the wild

`mlx_lm.generate.stream_generate` times a generation with `time.perf_counter` in the
same two phases: the prompt time ends when the first token arrives (the generator
underneath it has evaluated the prefill by then), and `prompt_tps` and
`generation_tps` are computed from the two intervals. `mlx_lm.benchmark` performs one
warm-up generation before its timed trials, for the reason given above. Two of its
choices differ from this problem's, and both are worth knowing when you read its
output. Its generation rate divides *all* generated tokens, the first included, by
the time since the first token arrived, so it is the rate of this problem times
$(n + 1) / n$: negligible for a few hundred tokens, a factor of two for two. And it
reports the mean of the per-trial rates, not a median. When you compare two numbers
from two tools, check three things before the numbers themselves: what was evaluated
before each clock reading, whether the first run was discarded, and which tokens were
divided by which interval.
