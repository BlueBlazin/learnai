## The idea

An MLX array is a node in a graph. `model(tokens)`, `mx.argmax`, and `+` only add
nodes; `mx.eval(x)` computes `x` together with every unevaluated node `x` depends on,
and afterwards those nodes hold their values and forget their inputs. An evaluated
array is a leaf: building on it starts a new, small graph.

So an evaluation is a *cut*. Everything a step hands to the next step, or to the
caller, should cross the cut as a computed array. In this loop three things outlive a
step:

| what | who reads it next | computed by `mx.eval(token)` alone? |
|---|---|---|
| the token | the next model call, the caller | yes |
| the cache | the next model call | yes: the token is computed from the logits, which are computed from every cache array, so the cache is an ancestor of the token |
| the running log-probability | the next step's `+`, the caller | **no**: nothing the token depends on depends on it |

The third row is the trap. An accumulator that hangs off the side of the graph is not
touched by evaluating the main result. It grows by three operations per step (the
indexing `logprobs[token]` and the addition), and, worse, each of them keeps that
step's log-probabilities alive. After 200 tokens the "scalar" is a chain of 600
pending operations.

One `mx.eval` with all three arguments evaluates one graph. Passing the cache is
redundant for this model and costs nothing (an evaluated array is skipped); it is the
habit worth having, because a cache with parts the logits do not read (a rotating
buffer's write position, usage statistics) has the same problem as the accumulator.

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

The prefill needs no evaluation of its own: its graph is part of what the first
`mx.eval` computes. `token.reshape(1)` is lazy, but it is one operation on a leaf;
it joins the next step's graph. `logprobs[token]` indexes with an array, so no Python
int is ever needed.

## What the placements cost

Measured on this machine (M1 Max, MLX 0.32.4), 200 tokens from a 4-token prompt, best
of four runs. "Tiny" is the default `MLXTinyLM`; "larger" has 6 layers of width 256 and
a vocabulary of 8192.

| placement | tiny | larger | pending operations handed to the last model call |
|---|---|---|---|
| one `mx.eval(token, logprob, cache)` per token | 141 ms | 467 ms | 1 |
| two `.item()` reads per token | 205 ms | 508 ms | 0 |
| two `mx.eval` calls per token | 252 ms | 510 ms | 1 |
| nothing in the loop, one `mx.eval` at the end | 103 ms | 299 ms | about 243 per step: 48,000 |

Two things to read from it. Every extra evaluation has a fixed cost (scheduling a
graph, a round trip to the GPU, waiting for it): 0.2 to 0.5 ms per token in this
table, which is comparable to the tiny model's whole step (0.7 ms) and still close to
a tenth of the larger one's (2.3 ms). And deferring
everything is *faster* in total, because the fixed cost is paid once. It is still the
wrong program: no token exists until the last one does, so nothing can be streamed, no
stop condition can be tested, no sampler can look at a value, and the graph (and
everything it keeps alive) grows with the length of the generation. The MLX
documentation's advice is the same trade: evaluate at each iteration of the outer
loop, neither per operation nor never. In a generation loop the outer iteration is
the token.

## Common mistakes

- **Reading with `.item()` inside the loop.** `token.item()` evaluates the token (and
  the cache); `logprobs[token].item()` then evaluates again. Both are evaluations you
  did not write down, two per step, and the token leaves the GPU only to be sent back.
- **Evaluating only at the end.** Right answer, but the second model call is handed a
  graph of 240 pending operations, the eighth 1,698, the sixtieth 14,334.
- **Evaluating the token (and cache) but not the accumulator.** The loop looks
  disciplined and the model always receives evaluated inputs, but `logprob` is
  returned as an unevaluated chain through every step. The caller's first `.item()`
  pays for all of it at once, and until then every step's logits stay allocated.
- **Splitting the step's evaluation**: `mx.eval(token)` then `mx.eval(logprob, cache)`.
  Two graphs are scheduled where one would do.
- **Evaluating the prefill separately** "to warm up". It is one more evaluation than
  tokens; the first step's `mx.eval` would have computed the prefill anyway.
- **Feeding the last token.** In a lazy framework the extra forward pass is never
  computed if nothing evaluates it, but it is still built, and the moment someone adds
  `logits` to the `mx.eval` call it becomes a wasted pass per generation.
- **Summing logits instead of log-probabilities.** The score needs the `logsumexp`
  normalization; raw logits are not comparable across steps.

## Complexity

$n$ evaluations, $n$ forward calls, $P + n - 1$ tokens through the model. The graph
alive at any time is one step: a few hundred operations for this model, independent
of $n$.

## In the wild

`mlx_lm.generate_step` is this loop with one refinement, the subject of
`mlx-streaming-generate`: it calls `mx.async_eval` on step $t + 1$ before reading the
token of step $t$, so the GPU is never idle while Python handles a token. The cut is
in the same place: one evaluation per token, covering the token and the
log-probabilities. `mx.export_to_dot(file, x)` is the tool the judge uses to count
pending operations, and it is worth knowing when a loop is mysteriously slow: export
the graph of what you are about to evaluate and look at its size.
