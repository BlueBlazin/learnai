## The idea

`mx.eval(x)` schedules the graph behind `x` and waits for it. `mx.async_eval(x)` only
schedules: it returns while the GPU is still working, and `x` (with everything it
depends on) is from then on a promise. Reading a promise, with `.item()` or anything
else that needs the number, waits for that array and no longer.

That splits "evaluate" into two moments you can place separately: when the work
starts, and when you wait for it. A generation loop wants them far apart. The value of
token $t$ is needed by Python (to detokenize, to print, to test a stop condition). It
is *not* needed to build step $t + 1$: the model takes the token as an array, and an
array that is still being computed is a perfectly good input to a graph. So:

1. schedule step $t + 1$, built from the array of token $t$;
2. now read token $t$. This waits until step $t$ is done, which it mostly is;
3. detokenize, yield, let the consumer do its work. All of this time, step $t + 1$
   is running.

When the loop comes back for token $t + 1$, it is usually ready, and the first thing
the loop does is schedule step $t + 2$. The GPU's queue is never empty, and Python
never waits for something it could have started earlier.

Reverse steps 1 and 2 and every line still runs, every token is still right, and the
GPU sits idle through step 3, once per token.

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

Three details.

**What goes into `async_eval`.** Everything the consumer or the next step will touch
that the token does not already depend on. The cache is an ancestor of the token (the
logits read every key and value), so scheduling the token schedules the cache. The
log-probability is a *descendant* of the token (`logprobs[token]`), so it must be
named, or it is yielded as a two-operation graph that the consumer's first read
evaluates synchronously, one extra round trip per token. This is the accumulator of
`mlx-lazy-eval-placement` again: evaluating the main result does not evaluate what
hangs off its side.

**The last token.** There is no step $n$. The loop knows it is at the last token and
schedules nothing more. A stop token is different: you cannot know that token $t$ is
the end of the text until you have read it, and by then step $t + 1$ is in flight.
That one wasted step is the price of the overlap, and every real loop pays it; what
it must not do is pay it at a boundary it knew in advance.

**It is a generator.** The body does not run until the consumer asks for the first
pair, and it pauses at every `yield`. The work the consumer does between two pairs is
exactly the time the scheduled step uses.

## What the overlap is worth

Measured on this machine (M1 Max, MLX 0.32.4), 100 tokens from a 4-token prompt, best
of four runs, milliseconds per token. "Larger" is an `MLXTinyLM` with 6 layers of
width 256 and a vocabulary of 8192; "wide" has 8 layers of width 1024. The Python
work per token is simulated by a `detok` that spins for the given time.

| model | Python work per token | `mx.eval`, then read | `mx.async_eval` of the next step, then read |
|---|---|---|---|
| tiny | none | 0.73 | 0.49 |
| larger | none | 2.30 | 1.30 |
| larger | 0.5 ms | 2.76 | 1.72 |
| wide | none | 3.15 | 2.21 |
| wide | 0.5 ms | 4.36 | 2.25 |
| wide | 2.0 ms | 5.82 | 3.74 |

Two effects add up. Even with no Python work at all, the asynchronous loop is faster,
because building the graph of step $t + 1$ (a few hundred microseconds of Python for
these models) happens while the GPU computes step $t$, and not after it. And Python
work up to the length of a step is free: the wide model takes 2.2 ms per token
whether the detokenizer needs nothing or half a millisecond, where the blocking loop
pays for it in full. Once the Python work is longer than a step (the last row), the
loop is limited by Python and the step hides inside it.

## Common mistakes

- **`mx.eval`, then read.** The loop of `mlx-lazy-eval-placement` with a `yield`
  added. Correct and serial: the left column of the table.
- **Scheduling after the read.** `async_eval` is there, in the wrong place: the token
  is read and detokenized first, and only then is the next step built and scheduled.
  Asynchronous evaluation of a step nobody overlaps with anything is a blocking
  evaluation with extra steps. The same happens, less visibly, when the loop converts
  the token to an int to feed it back (`mx.array([token.item()])`): the read is forced
  before the next step can even be built.
- **A step after the last token.** Unconditionally scheduling "the next step" builds
  and computes a forward pass whose token is never yielded. With `async_eval` the work
  is really done: it was handed to the GPU, not left in a graph.
- **Scheduling the token and not the log-probability.** The token is evaluated in the
  background and the log-probability is handed out lazy. Whoever reads it pays a
  synchronous evaluation, and until then it keeps that step's logits alive.
- **Collecting, then yielding.** A function that runs the whole loop and yields the
  pairs from a list at the end has the right tokens, the right evaluation order, and
  a time to first token equal to the time to last token.
- **Advancing before yielding.** Moving to the next step first and yielding its token
  drops the token chosen from the prompt: the stream starts with the second token.

## Complexity

$n$ scheduled evaluations, $n$ reads, $n$ forward calls, $P + n - 1$ tokens through
the model. At most two steps exist at a time: the one being read and the one in
flight.

## In the wild

`mlx_lm.generate.generate_step` is this loop with a sampler in place of the argmax and
logits processors in front of it: it calls `mx.async_eval(next_y, next_logprobs)` and
then yields `y.item()`. Its cache objects are updated in place by the model, so the
cache state is part of what the token depends on, as here. It also treats the token
limit the way it must treat a stop token: when the limit is reached, the step after
the last token has already been scheduled, one forward pass more than this problem
allows. `mx.async_eval` is marked
experimental in the MLX documentation; the pattern around it is the stable part.
`EvalSpy`, the kit object the judge uses to see the order of evaluations, cannot see
an evaluation forced by `if x:` or by an f-string, so this problem also checks the
overlap from the other side: the detokenizer it hands you notes how many times the
model has been called each time it is asked for a piece of text.
