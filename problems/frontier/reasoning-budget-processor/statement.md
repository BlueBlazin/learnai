Write the logits processor that caps how long a reasoning model may think. It runs
once per generated token, keeps track of where generation is (before the thinking
span, inside it, closing it, or in the answer), and when the thinking-token budget
is used up it takes the choice away from the model: the next tokens are the
end-of-thinking sequence, one per step, and then the answer proceeds. The processor
sees one token at a time, so everything it knows it has to carry in its state.

## Specification

```python
def budget_step(state, token, logits, cfg): ...  # -> (state, logits)
```

The generation loop calls `budget_step` once before each token is chosen:

```python
state, token = None, None
while generating:
    raw = model(...)  # logits for the next token, shape (V,)
    state, logits = budget_step(state, token, raw, cfg)
    token = choose(logits)  # any token whose logit is not -inf
```

- `state`: `None` on the first call; afterwards whatever the previous call
  returned. Its form is yours to choose; the caller only passes it back.
- `token`: `None` on the first call; afterwards the token (a Python int) that was
  chosen from the logits the previous call returned.
- `logits`: the model's raw logits for the next token, shape `(V,)`, finite.
- `cfg`: a dict.

  | key | meaning |
  |---|---|
  | `"start_id"` | the start-of-thinking token |
  | `"end_id"` | the end-of-thinking token |
  | `"forced"` | the **end sequence**: a list of one or more token ids, forced when the budget is used up. Its last element is `end_id`, which occurs nowhere else in it; `start_id` does not occur in it. |
  | `"budget"` | the largest number of thinking tokens, an int $\ge 0$ |
  | `"answer_mask"` | `None`, or a bool tensor of shape `(V,)`: the tokens the answer may contain (a stand-in for a structured-output constraint) |

- Returns `(state, logits)`: the new state, and a float32 tensor of shape `(V,)` in
  which every token that may not come next is `-inf` and every other entry is the
  raw logit, unchanged.

**Phases.** A generation is in one of four phases. Each call first moves the phase
forward by `token`, then processes the logits for the phase it is now in.

| phase | when | tokens that may come next |
|---|---|---|
| *start* | the first call: nothing has been generated | all, if `answer_mask` is `None`; otherwise the tokens of `answer_mask` and `start_id` |
| *thinking* | inside the span, with $u$ thinking tokens so far, $u <$ `budget` | all (the answer mask does **not** apply) |
| *forcing* | the budget is used up; $i$ tokens of the end sequence have been generated | only `forced[i]` |
| *answer* | the span is closed, or was never opened | all, if `answer_mask` is `None`; otherwise the tokens of `answer_mask` |

**Transitions**, by the token that was chosen:

1. From *start*: `start_id` opens the span (*thinking* with $u = 0$); any other
   token is the first token of the answer (*answer*). A span can only open with
   the first generated token.
2. From *thinking*: `end_id` closes the span (*answer*): the model stopped on its
   own, and nothing is forced. Any other token, `start_id` included, is a thinking
   token: $u \leftarrow u + 1$.
3. Whenever the phase is *thinking* and $u =$ `budget`, it becomes *forcing* with
   $i = 0$ at once, before the logits are processed. With `budget = 0` this
   happens in the call that receives `start_id`.
4. From *forcing*: $i \leftarrow i + 1$; when $i$ reaches `len(forced)` the phase is
   *answer*.
5. *answer* is final. A `start_id` in the answer is an ordinary token.

So the thinking tokens are the tokens after `start_id` and before the end sequence
or the model's own `end_id`; there are never more than `budget` of them, and the
tokens of the end sequence do not count.

## Shapes

| name | shape | dtype |
|---|---|---|
| `logits` (argument) | `(V,)` | float32 |
| `cfg["answer_mask"]` | `(V,)` or `None` | bool |
| `logits` (returned) | `(V,)` | float32 |

## Example

$V = 8$, `start_id = 6`, `end_id = 7`, `forced = [5, 7]` (token 5 stands for a short
"time to answer" message), `budget = 2`, and an answer mask that allows tokens 0, 1,
and 2. The loop chooses the largest remaining logit.

| call | `token` | phase after moving | raw logits | may come next | chosen |
|---|---|---|---|---|---|
| 1 | `None` | start | `[0.5, 0.1, 0.3, 2.0, 0.2, 0.0, 1.5, 0.4]` | 0, 1, 2, 6 | 6 |
| 2 | 6 | thinking, $u = 0$ | `[0.1, 0.2, 0.0, 0.3, 1.2, 0.5, 0.4, 0.6]` | all | 4 |
| 3 | 4 | thinking, $u = 1$ | `[0.3, 0.1, 0.2, 1.4, 0.5, 0.0, 0.6, 0.9]` | all | 3 |
| 4 | 3 | $u = 2$: forcing, $i = 0$ | `[1.0, 0.2, 0.1, 0.4, 0.8, -0.5, 0.3, 0.6]` | 5 | 5 |
| 5 | 5 | forcing, $i = 1$ | `[0.2, 0.9, 0.1, 0.3, 0.4, 0.5, 0.6, 0.0]` | 7 | 7 |
| 6 | 7 | answer | `[0.1, 0.4, 0.7, 1.1, 0.9, 0.2, 1.3, 0.5]` | 0, 1, 2 | 2 |
| 7 | 2 | answer | `[0.8, 0.3, 0.2, 0.1, 0.6, 0.4, 0.0, 1.0]` | 0, 1, 2 | 0 |

Call 1 returns `[0.5, 0.1, 0.3, -inf, -inf, -inf, 1.5, -inf]`: the model's favorite,
token 3, is not an answer token, and of what is left it prefers to think. Calls 2
and 3 return the raw logits: tokens 4 and 3 are not answer tokens, and inside the
span that does not matter. Call 4 receives the second thinking token, so the budget
is used up, and returns `[-inf, -inf, -inf, -inf, -inf, -0.5, -inf, -inf]`: token 5
keeps its logit, low as it is. Call 6 is the first call of the answer, and the mask
is in force from that call on: the raw favorite there is token 6, which would have
been another `start_id`.

Had the model chosen token 7 in call 3, call 4 would already be in the answer phase,
with nothing forced. With `budget = 0`, call 2 would return the row that only
allows token 5.

## Constraints

- $4 \le V \le 64$, $0 \le$ `budget`, `1 <= len(forced) <= 4`. The raw logits are
  finite.
- The returned logits are compared with the reference's: the same entries must be
  `-inf`, and the others must equal the raw logits (`rtol = 1e-4`, `atol = 1e-6`).
- The loop only ever chooses a token whose returned logit is not `-inf`. In most
  cases the test chooses the tokens; a wrong row is reported at the first call that
  produced one.
- **Cost.** In the model-run cases your processor steers a
  [`TinyLM`](../../../python/src/whetstone/kit/tinylm.py) by itself (the next token
  is the argmax of the row you return). The run must open the span, let at most
  `budget` thinking tokens through (reported as `thinking_tokens`), close the span
  with the complete end sequence, and keep the answer inside the mask.
- Do not modify `cfg` or the `logits` argument.

## Why this matters

A reasoning model decides for itself how long to think, and the serving engine
pays for it: thinking tokens occupy a batch slot, count against `max_tokens`, and
delay the first token of the answer a user is waiting for. A thinking budget turns
that into a bound the operator controls, and it is enforced exactly here, between
the model's logits and the sampler. The details are what production bugs are made
of. Force only the first token of a multi-token end marker and the span never
closes, so the parser files the whole answer under reasoning. Count the wrong
tokens and the budget is off by one on every request. Apply the JSON grammar while
the model is still thinking and it has to reason in JSON; forget to apply it on the
first answer token and the "guaranteed" structure starts with prose.
