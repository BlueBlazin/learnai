Implement a logits processor that caps how many tokens a reasoning model may spend thinking. It runs once before each generated token, tracks the current generation phase, forces a fixed end-of-thinking sequence once the budget is used up, and applies an optional answer mask outside the thinking span.

## Specification

```python
def budget_step(state, token, logits, cfg): ...  # -> (state, logits)
```

The generation loop calls `budget_step` once before each token is chosen, where $V$ is the vocabulary size:

```python
state, token = None, None
while generating:
    raw = model(...)  # next-token logits, shape (V,)
    state, logits = budget_step(state, token, raw, cfg)
    token = choose(logits)  # any token whose logit is not -inf
```

- `state`: `None` on the first call, and whatever the previous call returned thereafter. Its structure is up to you, since the caller only passes it back unchanged.
- `token`: `None` on the first call. On every subsequent call, it is the Python `int` token chosen from the previous call's returned logits.
- `logits`: the model's finite raw logits for the next token.
- `cfg`: a configuration dictionary with the following keys:

  | key | meaning |
  |---|---|
  | `"start_id"` | start-of-thinking token ID |
  | `"end_id"` | end-of-thinking token ID |
  | `"forced"` | the **end sequence**, a non-empty list of token IDs forced when the budget is exhausted. Its last element is `end_id`, which does not appear earlier in the list, and `start_id` does not appear in it at all. |
  | `"budget"` | maximum number of thinking tokens, an integer $\ge 0$ |
  | `"answer_mask"` | `None`, or a boolean tensor marking which tokens the answer may contain (a stand-in for a structured-output constraint) |

- Returns `(state, logits)`, where `state` is the updated state and `logits` has `-inf` for every disallowed token and the original raw logit unchanged for every allowed token.

**Phases.** Generation is always in one of four phases. Each call first advances the phase using `token`, and then masks `logits` according to the resulting phase. Let $u$ be the number of thinking tokens generated so far, and let $i$ be the number of tokens from `forced` generated so far.

| phase | when | allowed next tokens |
|---|---|---|
| *start* | first call, before any token is generated | all tokens if `answer_mask` is `None`, otherwise the tokens in `answer_mask` plus `start_id` |
| *thinking* | inside the thinking span with $u <$ `budget` | all tokens (`answer_mask` does **not** apply) |
| *forcing* | budget is exhausted, with $i$ tokens of `forced` already generated | only `forced[i]` |
| *answer* | thinking span is closed, or was never opened | all tokens if `answer_mask` is `None`, otherwise the tokens in `answer_mask` |

**Transitions.** On any call after the first, update the phase using `token`:

1. From *start*, `start_id` opens the thinking span and enters *thinking* with $u = 0$. Any other token is the first token of the answer and enters *answer*. A thinking span can only open on the first generated token.
2. From *thinking*, `end_id` closes the span and enters *answer* without forcing anything. Any other token, including `start_id`, counts as a thinking token and increments $u \leftarrow u + 1$.
3. Whenever the phase is *thinking* and $u =$ `budget`, transition immediately to *forcing* with $i = 0$ before masking `logits`. When `budget = 0`, this happens on the same call that receives `start_id`.
4. From *forcing*, increment $i \leftarrow i + 1$. When $i$ reaches `len(forced)`, transition to *answer*.
5. *answer* is terminal. A `start_id` appearing in the answer is treated as an ordinary token.

Under these rules, thinking tokens are the tokens generated after `start_id` and before either `forced` or the model's own `end_id`. There are never more than `budget` thinking tokens, and tokens in `forced` do not count toward the budget.

## Shapes

| name | shape | dtype |
|---|---|---|
| `logits` (argument) | `(V,)` | float32 |
| `cfg["answer_mask"]` | `(V,)` or `None` | bool |
| `logits` (returned) | `(V,)` | float32 |

## Example

Suppose $V = 8$, `start_id = 6`, `end_id = 7`, `forced = [5, 7]` (where token 5 represents a short cutoff message), `budget = 2`, and `answer_mask` allows tokens 0, 1, and 2. Assume the generation loop picks the largest non-`-inf` logit.

| call | `token` | phase after transition | raw logits | allowed next | chosen |
|---|---|---|---|---|---|
| 1 | `None` | start | `[0.5, 0.1, 0.3, 2.0, 0.2, 0.0, 1.5, 0.4]` | 0, 1, 2, 6 | 6 |
| 2 | 6 | thinking, $u = 0$ | `[0.1, 0.2, 0.0, 0.3, 1.2, 0.5, 0.4, 0.6]` | all | 4 |
| 3 | 4 | thinking, $u = 1$ | `[0.3, 0.1, 0.2, 1.4, 0.5, 0.0, 0.6, 0.9]` | all | 3 |
| 4 | 3 | $u = 2 \rightarrow$ forcing, $i = 0$ | `[1.0, 0.2, 0.1, 0.4, 0.8, -0.5, 0.3, 0.6]` | 5 | 5 |
| 5 | 5 | forcing, $i = 1$ | `[0.2, 0.9, 0.1, 0.3, 0.4, 0.5, 0.6, 0.0]` | 7 | 7 |
| 6 | 7 | answer | `[0.1, 0.4, 0.7, 1.1, 0.9, 0.2, 1.3, 0.5]` | 0, 1, 2 | 2 |
| 7 | 2 | answer | `[0.8, 0.3, 0.2, 0.1, 0.6, 0.4, 0.0, 1.0]` | 0, 1, 2 | 0 |

Call 1 returns `[0.5, 0.1, 0.3, -inf, -inf, -inf, 1.5, -inf]`. Token 3 has the highest raw logit, but it is masked out because it is neither in `answer_mask` nor `start_id`, so the loop chooses token 6 to start thinking. Calls 2 and 3 return the raw logits unchanged because `answer_mask` is suspended inside the thinking span, letting tokens 4 and 3 through. Call 4 receives the second thinking token, which exhausts the budget, so it returns `[-inf, -inf, -inf, -inf, -inf, -0.5, -inf, -inf]`, preserving token 5's raw logit. Call 6 is the first answer step, where `answer_mask` takes effect again and blocks token 6 even though it has the highest raw logit.

If the model had chosen token 7 on call 3 instead, call 4 would have entered *answer* directly without forcing anything. With `budget = 0`, call 2 would immediately enter *forcing* and allow only token 5.

## Constraints

- $4 \le V \le 64$, $0 \le$ `budget`, and `1 <= len(forced) <= 4`. Raw logits are finite.
- Returned logits are checked against the reference processor. The exact same positions must be `-inf`, and all other positions must match the raw logits within `rtol = 1e-4` and `atol = 1e-6`.
- The generation loop only ever chooses a token whose returned logit is not `-inf`. In most test cases, the test harness supplies the chosen tokens and reports a failure on the first call whose output is wrong.
- **Cost.** In the model-run test cases, your processor steers a [`TinyLM`](../../../python/src/learnai/kit/tinylm.py) by itself, with the next token chosen as the argmax of your returned logits. The run must open the thinking span, allow at most `budget` thinking tokens (reported as `thinking_tokens`), close the span with the complete `forced` sequence, and keep all answer tokens inside `answer_mask`.
- Do not modify `cfg` or the `logits` argument in place.

## Why this matters

Reasoning models choose how long to think, and every thinking token consumes a serving batch slot, counts against `max_tokens`, and delays the first answer token. A thinking-token budget gives operators a hard upper bound enforced right between the model's logits and the sampler.

Small bugs in this processor cause silent production failures. Forcing only the first token of a multi-token end marker leaves the thinking span unclosed, so the output parser classifies the entire answer as reasoning. Miscounting tokens makes every request off by one. Applying a JSON grammar during thinking forces the model to reason in JSON, while applying it one token too late lets unconstrained prose slip into the start of a structured output.
