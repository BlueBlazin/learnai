Write a benchmark harness for an MLX language model that runs a prompt and a short generation several times, then reports prompt tokens per second and generation tokens per second. In a lazy framework like MLX, a model call returns in microseconds because it only builds a computation graph, and the actual compute happens wherever that graph is finally evaluated. The judge supplies a simulated clock that models this exact behavior, so any misplaced evaluation produces the wrong throughput numbers.

## Specification

```python
def benchmark(model, prompt, n_tokens, n_warmup, n_runs, clock): ...
```

`model` behaves like [`MLXTinyLM`](../../../python/src/learnai/kit/mlxlm.py) from `mlx-lazy-eval-placement`. Calling `logits, cache = model(tokens)` processes a 1-D integer array of tokens from scratch, and `logits, cache = model(tokens, cache)` continues from an existing KV cache. Both calls return **lazy** arrays, and `logits[-1]` contains the next-token scores. `clock()` returns the current time in seconds as a float.

Let $P = \text{len}(\text{prompt})$ be the number of prompt tokens. **One run** performs greedy generation with three clock readings:

1. Read the clock to get $t_0$.
2. **Prompt phase.** Call `model` on the full prompt, take the argmax of `logits[-1]` as the first generated token, and evaluate it so the prefill completes.
3. Read the clock to get $t_1$.
4. **Generation phase.** Repeat `n_tokens` times: pass the latest token and `cache` to `model`, take the argmax of `logits[-1]` as the next token, and evaluate it before moving to the next step.
5. Read the clock to get $t_2$.

For each run, the prompt time is $t_1 - t_0$ (covering $P$ prompt tokens) and the generation time is $t_2 - t_1$ (covering `n_tokens` decode steps). Each run makes $1 + \text{n\_tokens}$ calls to `model` and produces $\text{n\_tokens} + 1$ tokens, with the first token produced during the prompt phase.

`benchmark` runs `n_warmup` complete warm-up runs and discards their timings, then runs `n_runs` measured runs and keeps their timings. Every warm-up run must execute and evaluate just like a measured run. Return a dict with exactly these keys:

| key | value |
|---|---|
| `"prompt_tps"` | $P \,/\, \operatorname{median}(\text{prompt times of the measured runs})$, as a Python float |
| `"generation_tps"` | $\text{n\_tokens} \,/\, \operatorname{median}(\text{generation times of the measured runs})$, as a Python float |
| `"tokens"` | the $\text{n\_tokens} + 1$ tokens from a run, as a list of Python ints (every run produces the same sequence) |

For an even number of measured runs, the median is the mean of the two middle values.

**Judge rules.**

- Evaluate using `mx.eval` (or by reading a scalar value). Do not use `mx.async_eval`, because it returns before the computation finishes.
- Run exactly `n_warmup + n_runs` full runs. `model.counters.forward_calls` must equal $(\text{n\_warmup} + \text{n\_runs}) \cdot (1 + \text{n\_tokens})$, and `tokens_processed` must equal $(\text{n\_warmup} + \text{n\_runs}) \cdot (P + \text{n\_tokens})$.
- Do not let a lazy graph span across steps. Whenever `model` is called, at most 8 operations may be pending behind its arguments (the same rule as `mlx-lazy-eval-placement`, where reshaping an evaluated token counts as 1).

**The simulated clock.** In the tests, `clock` is a deterministic simulator rather than wall-clock time. It advances in two ways:

- **Graph building.** Each call to `model` immediately advances the clock by a small fixed cost for constructing the graph.
- **Computation.** Each call to `model` also has a much larger compute cost, which is **not** charged when `model` is called. Instead, it is charged the first time `clock()` is read after that call's output has actually been evaluated. Unevaluated work adds zero time, and late evaluation is billed to whichever interval ends at the next clock reading.

Earlier computed runs are slower than later ones (simulating kernel compilation and buffer allocation), and occasional runs have extra latency. Because the simulator is deterministic, a correct harness produces exact throughput values.

## Shapes

| name | type |
|---|---|
| `prompt` | list of $P \ge 1$ Python ints |
| `n_tokens` | int ($\ge 1$), decode steps per run |
| `n_warmup` | int ($\ge 0$) |
| `n_runs` | int ($\ge 1$) |
| `clock` | callable returning float seconds |
| result | dict with two Python floats and a list of $\text{n\_tokens} + 1$ Python ints |

## Example

```python
benchmark(model, [0, 17, 42, 5], n_tokens=8, n_warmup=1, n_runs=3, clock=clock)
# {"prompt_tps": 493.83, "generation_tps": 243.90,
#  "tokens": [28, 34, 31, 15, 1, 48, 46, 5, 7]}
```

On this simulated machine, building a graph costs 0.1 ms per model call, prefill computes in 2 ms per prompt token, and each decode step computes in 4 ms. The first computed run is 5 times slower, and the third computed run is 1.5 times slower.

| run | note | prompt time | generation time |
|---|---|---|---|
| 0 | warm-up, cold (discarded) | $0.1 + 4 \cdot 2 \cdot 5 = 40.1$ ms | $8 \cdot (0.1 + 4 \cdot 5) = 160.8$ ms |
| 1 | measured | $0.1 + 4 \cdot 2 = 8.1$ ms | $8 \cdot (0.1 + 4) = 32.8$ ms |
| 2 | measured, slow outlier | $0.1 + 4 \cdot 2 \cdot 1.5 = 12.1$ ms | $8 \cdot (0.1 + 6) = 48.8$ ms |
| 3 | measured | $8.1$ ms | $32.8$ ms |

The medians across the three measured runs are 8.1 ms for prompt processing and 32.8 ms for generation. Dividing token counts by those medians gives

$$
\text{prompt\_tps} = \frac{4}{0.0081} = 493.83,
\qquad
\text{generation\_tps} = \frac{8}{0.0328} = 243.90 .
$$

A harness that reads the clock without evaluating measures only graph construction. It records 0.1 ms for the prompt and 0.8 ms for generation, reporting 40,000 and 10,000 tokens per second instead.

## Constraints

- $1 \le P \le 64$, $1 \le \text{n\_tokens} \le 32$, $0 \le \text{n\_warmup} \le 4$, and $1 \le \text{n\_runs} \le 8$.
- Returned tokens must match greedy decoding without a cache exactly. All test models keep the top two logits separated by more than `1e-3`.
- `prompt_tps` and `generation_tps` are checked against the exact simulated values with a relative tolerance of `1e-6`.
- The run count and the 8-operation pending limit are enforced exactly as specified above.
- One test case runs on the Apple GPU with `time.perf_counter` and a real model, checking only that the rates are finite and positive and that the tokens match.

## Why this matters

Published tokens-per-second numbers on Apple silicon come from loops like this one, and lazy evaluation makes it easy to produce misleading results that still look plausible. Timing a model call without evaluating it measures only graph construction. On an M1 Max, a 256-token prefill that actually takes 17 ms appears to finish in 0.4 ms, reporting 700,000 tokens per second. Forgetting to evaluate before the middle clock reading bills the entire prefill to the first decode step, making time to first token look instant and generation look slow.

Skipping warm-up or running it lazily leaves kernel compilation and initial buffer allocations inside the first measured run, which doubled the 256-token prefill time to 35 ms on the same machine. Taking a mean over a few runs lets a single OS scheduling hiccup skew the reported speed. `mlx_lm.benchmark` discards a warm-up run and times evaluated work in the same two phases, though this problem uses stricter definitions for the median and token count.
