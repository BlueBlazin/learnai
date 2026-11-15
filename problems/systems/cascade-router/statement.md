Most questions do not need the largest model. A cascade asks the cheapest model
first, looks at how confident it was, and pays for a larger model only when the
answer does not clear a bar. Write the loop: one call per stage, a confidence
computed from the token log-probabilities, a threshold per stage, and a budget that
is checked **before** money is spent, not after.

## Specification

```python
def cascade(prompt: str, models: list, thresholds: list[float], budget: float) -> dict: ...
```

**The models.** `models` is a list of $n \ge 1$ kit `ScriptedChatModel` objects in
the order they are tried, cheapest first. For a model `m`:

- `m.name` is its name, and `m.cost` the price of **one call**, known before
  calling (a non-negative integer, in whatever unit the budget uses). Costs do not
  decrease along the list.
- `m.complete(messages)` makes one call and returns the assistant message

  ```python
  {"role": "assistant", "content": "Paris", "logprobs": [-0.1, -0.5, -0.6]}
  ```

  `content` is the answer and `logprobs` holds the log-probability of each token of
  the answer, in order (at least one, all $\le 0$).

**Confidence.** The confidence of a reply with token log-probabilities
$\ell_1, \dots, \ell_T$ is their mean,

$$
c = \frac{1}{T} \sum_{t=1}^{T} \ell_t ,
$$

the logarithm of the geometric mean of the token probabilities. It does not depend
on how long the answer is.

**The cascade.** `thresholds` has $n - 1$ entries, one for every model but the last.
Start with `spent = 0` and go through the models in order. At model $i$ (from 0):

1. **Budget.** If `spent + models[i].cost > budget`, the cascade stops here: model
   $i$ is not called, and neither is any later model.
2. **Call.** Otherwise call it once with the single message
   `[{"role": "user", "content": prompt}]`, and add its cost to `spent`.
3. **Decide.** If $i$ is the last model, its answer is accepted. Otherwise it is
   accepted when $c \ge$ `thresholds[i]`. An accepted answer ends the cascade; a
   rejected one sends the prompt on to model $i + 1$.

**The result** is a dict with exactly these keys:

| key | value |
|---|---|
| `"answer"` | the `content` of the **last model that was called**, or `None` if no model was called |
| `"model"` | that model's `name`, or `None` |
| `"confidence"` | the confidence of that reply (a float), or `None` |
| `"cost"` | `spent`: the sum of the costs of the models that were called |
| `"accepted"` | `True` if the answer was accepted in step 3, `False` if the budget stopped the cascade |

When the budget stops the cascade, the answer is the one of the last model called,
whatever the confidences of the earlier replies were: confidences of different
models are not comparable, and a later model is a stronger one.

Every model is called at most once, in order, and no model is called after an
answer has been accepted.

## Example

```python
small  # cost 1,  replies "Lyon"  with logprobs [-0.1, -0.5, -0.6]
medium  # cost 5,  replies "Paris" with logprobs [-0.25, -0.75]
large  # cost 20, replies "Paris" with logprobs [-0.01, -0.03]

cascade("Capital of France?", [small, medium, large], [-0.2, -0.5], budget=30)
# {"answer": "Paris", "model": "medium", "confidence": -0.5, "cost": 6, "accepted": True}
```

- `small`: `0 + 1 <= 30`, so it is called. Its confidence is
  $(-0.1 - 0.5 - 0.6) / 3 = -0.4$, below the threshold $-0.2$: rejected.
- `medium`: `1 + 5 <= 30`. Its confidence is $(-0.25 - 0.75) / 2 = -0.5$, which is
  not below the threshold $-0.5$: accepted. `large` is never called, and the
  request cost 6 instead of 20.

With `budget=5` the second call would bring the total to 6, so it is not made:

```python
cascade("Capital of France?", [small, medium, large], [-0.2, -0.5], budget=5)
# {"answer": "Lyon", "model": "small", "confidence": -0.4, "cost": 1, "accepted": False}
```

Note what the sum of the log-probabilities would have done: $-1.2$ for the
three-token answer against $-1.0$ for the two-token one. A sum punishes every
additional token, so long answers would always be escalated.

## Constraints

- $1 \le n \le 8$; `len(thresholds) == n - 1`; thresholds are $\le 0$; `budget`
  $\ge 0$. Costs and budgets are integers, so `cost` is compared exactly.
- `answer`, `model`, and `accepted` are compared exactly; `confidence` to an
  absolute tolerance of `1e-9`. Where a confidence equals its threshold, the
  log-probabilities are binary fractions, so the mean is exact however it is
  summed; otherwise confidences are at least `1e-6` away from their thresholds.
- **Cost.** The tests read each model's call counter and the messages it received:
  the models called, the number of calls (at most one each), and the total spend
  must be exactly what the rule above gives. A model that is called "to compare"
  is paid for.

## Why this matters

A request to a frontier model costs ten to a hundred times what the same request
costs on a small one, and for a large share of production traffic (classification,
extraction, short factual answers) the small model's answer is the same. A cascade
turns that into money: route everything to the cheap model and escalate the
fraction it is unsure about. The three ways it goes wrong are the three things this
problem checks. A confidence that depends on answer length escalates exactly the
requests with long answers, which are the expensive ones. A router that calls the
large model anyway, to compare, has the latency of the cheap path and the bill of
the expensive one. And a budget that is checked after the call has already been
exceeded by the time it says no.
