Write the benchmark harness for an MLX language model: run a prompt and a short
generation several times and report prompt tokens per second and generation tokens per
second. In a lazy framework the obvious harness measures nothing: a model call
returns in microseconds because it only builds a graph, and the time appears wherever
the graph is finally evaluated. The judge hands you a simulated clock on which exactly
that happens, so every misplaced evaluation shows up as a wrong number.

## Specification

```python
def benchmark(model, prompt, n_tokens, n_warmup, n_runs, clock): ...
```

`model` behaves like the [`MLXTinyLM`](../../../python/src/whetstone/kit/mlxlm.py) of
`mlx-lazy-eval-placement`: `logits, cache = model(tokens)` runs a 1-D integer array of
tokens from scratch, `logits, cache = model(tokens, cache)` continues a cache, and
both return **lazy** arrays; `logits[-1]` scores the next token. `clock()` returns the
current time in seconds as a float.

**One run** is a greedy generation with three clock readings:

1. read the clock: $t_0$;
2. *prompt phase*: call the model on the whole prompt, take the argmax of
   `logits[-1]` as the first token, and make sure it is computed;
3. read the clock: $t_1$;
4. *generation phase*: `n_tokens` times, feed the latest token with the cache, take
   the argmax as the next token, and make sure it is computed, one evaluation per
   step, as a generation loop does;
5. read the clock: $t_2$.

The run's prompt time is $t_1 - t_0$, for $P$ prompt tokens processed. Its generation
time is $t_2 - t_1$, for `n_tokens` decode steps. A run makes $1 + \text{n\_tokens}$
model calls and produces $\text{n\_tokens} + 1$ tokens, the first of them during the
prompt time.

`benchmark` performs `n_warmup` runs whose times are discarded, then `n_runs` runs
whose times are kept. A warm-up run is a complete run, computed like any other. It
returns a dict with exactly these keys:

| key | value |
|---|---|
| `"prompt_tps"` | $P \,/\, \operatorname{median}(\text{prompt times of the measured runs})$, a Python float |
| `"generation_tps"` | $\text{n\_tokens} \,/\, \operatorname{median}(\text{generation times of the measured runs})$, a Python float |
| `"tokens"` | the $\text{n\_tokens} + 1$ tokens of a run, as a list of Python ints (every run produces the same ones) |

The median of an even number of values is the mean of the two middle ones.

**Rules the judge checks.**

- Evaluate with `mx.eval` (or by reading a value). `mx.async_eval` is not accepted:
  it returns before the work is done.
- Exactly `n_warmup + n_runs` runs are made: `model.counters.forward_calls` is exactly
  $(\text{n\_warmup} + \text{n\_runs}) \cdot (1 + \text{n\_tokens})$ and
  `tokens_processed` exactly $(\text{n\_warmup} + \text{n\_runs}) \cdot (P + \text{n\_tokens})$.
- No lazy graph crosses a step: whenever the model is called, at most 8 operations
  are pending behind its arguments (the rule of `mlx-lazy-eval-placement`; reshaping an
  evaluated token is 1).

**The simulated clock.** In the tests, `clock` is not the wall clock, and time passes
on it in two ways only:

- *Building.* Every model call advances the clock at once by a small fixed amount,
  the cost of constructing a graph.
- *Computing.* Each model call also has a compute time, which is most of its cost.
  It is **not** charged when the model is called. It is charged the first time the
  clock is read after that call's result has actually been computed. Work that was
  never evaluated has taken no time; work evaluated late is charged to whichever
  interval ends at the next clock reading.

The runs that are computed first are slower than the later ones, as on a real machine
(kernels are compiled, buffers are allocated), and individual runs are slower now and
then. All of it is deterministic: for a correct harness the two rates are exact
numbers.

## Shapes

| name | type |
|---|---|
| `prompt` | list of $P \ge 1$ Python ints |
| `n_tokens` | int, $\ge 1$: decode steps per run |
| `n_warmup` | int, $\ge 0$ |
| `n_runs` | int, $\ge 1$ |
| `clock` | callable, `clock() -> float` (seconds) |
| result | dict: two Python floats and a list of $\text{n\_tokens} + 1$ Python ints |

## Example

```python
benchmark(model, [0, 17, 42, 5], n_tokens=8, n_warmup=1, n_runs=3, clock=clock)
# {"prompt_tps": 493.83, "generation_tps": 243.90,
#  "tokens": [28, 34, 31, 15, 1, 48, 46, 5, 7]}
```

On this simulated machine, building a graph costs 0.1 ms per model call, the prefill
computes in 2 ms per prompt token, and a decode step in 4 ms. The first computed run
is 5 times slower, and the third is 1.5 times slower.

| run | | prompt time | generation time |
|---|---|---|---|
| 0 | warm-up, cold: discarded | $0.1 + 4 \cdot 2 \cdot 5 = 40.1$ ms | $8 \cdot (0.1 + 4 \cdot 5) = 160.8$ ms |
| 1 | measured | $0.1 + 4 \cdot 2 = 8.1$ ms | $8 \cdot (0.1 + 4) = 32.8$ ms |
| 2 | measured, a slow one | $0.1 + 4 \cdot 2 \cdot 1.5 = 12.1$ ms | $8 \cdot (0.1 + 6) = 48.8$ ms |
| 3 | measured | $8.1$ ms | $32.8$ ms |

The medians of the measured runs are 8.1 ms and 32.8 ms, so

$$
\text{prompt\_tps} = \frac{4}{0.0081} = 493.83,
\qquad
\text{generation\_tps} = \frac{8}{0.0328} = 243.90 .
$$

A harness that reads the clock without evaluating sees only the building: a prompt
time of 0.1 ms and a generation time of 0.8 ms, which it reports as 40,000 and 10,000
tokens per second.

## Constraints

- $1 \le P \le 64$, $1 \le \text{n\_tokens} \le 32$, $0 \le \text{n\_warmup} \le 4$,
  $1 \le \text{n\_runs} \le 8$.
- The tokens are compared exactly with greedy decoding that uses no cache. The test
  models keep the two largest logits more than `1e-3` apart.
- `prompt_tps` and `generation_tps` are compared with the values the simulated costs
  imply, to a relative tolerance of `1e-6`.
- The run count and the pending-operation limit are exact; see the rules above.
- One case, on the Apple GPU, passes `time.perf_counter` and an ordinary model. It
  checks only that the rates are finite and positive and that the tokens are right.

## Why this matters

Every published tokens-per-second figure for a model on Apple silicon comes out of a
loop like this one, and a lazy framework makes it unusually easy to get a wrong
number that looks plausible or flattering. Time a model call without evaluating it
and the result is the cost of building a graph: on this machine a 256-token prefill
that takes 17 ms "runs" in 0.4 ms, a 700,000 tokens-per-second prompt rate. Forget to
evaluate before the middle clock reading and the prefill is billed to the first
generated token, so time to first token looks instant and decoding looks slow. Skip
the warm-up, or run it lazily, and the first measured run pays for kernel compilation
and fresh allocations: the same prefill took 35 ms on the first run of a process.
And a mean over five runs, one of which lost a time slice to another process, is a
measurement of that other process. `mlx_lm.benchmark` discards a warm-up generation
and times evaluated work in the same two phases; the definitions of this problem are
the stricter ones.
