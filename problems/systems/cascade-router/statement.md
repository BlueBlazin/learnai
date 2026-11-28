Most prompts do not need the largest, most expensive model. A model cascade tries the cheapest model first, checks how confident that model is in its reply, and escalates to a larger model only when the reply falls below a confidence threshold. Implement this routing loop so that each model is called at most once and the budget is checked before spending money.

## Specification

```python
def cascade(prompt: str, models: list, thresholds: list[float], budget: float) -> dict: ...
```

**Models.** `models` is a list of $n \ge 1$ kit `ScriptedChatModel` objects ordered from cheapest to most expensive. Costs do not decrease along the list. Each model `m` provides:

- `m.name`, a string identifying the model.
- `m.cost`, a non-negative integer giving the price of one call (known before calling, in the same units as `budget`).
- `m.complete(messages)`, which makes one call and returns an assistant message dictionary:

  ```python
  {"role": "assistant", "content": "Paris", "logprobs": [-0.1, -0.5, -0.6]}
  ```

  Here `content` is the answer string and `logprobs` is a non-empty list of token log-probabilities for that answer, in order, with every entry $\le 0$.

**Confidence.** Given a reply with $T \ge 1$ token log-probabilities $\ell_1, \dots, \ell_T$, its confidence $c$ is their mean:

$$
c = \frac{1}{T} \sum_{t=1}^{T} \ell_t
$$

This equals the logarithm of the geometric mean of the token probabilities. Taking the mean normalizes for length so longer answers are not penalized.

**Routing loop.** `thresholds` is a list of $n - 1$ floats, one for every model except the last. Start with `spent = 0` and iterate through `models` in order. At index $i$ (starting from $0$):

1. **Check budget.** If `spent + models[i].cost > budget`, stop immediately. Do not call model $i$ or any later model.
2. **Call model.** Otherwise, call `models[i].complete([{"role": "user", "content": prompt}])` once and add `models[i].cost` to `spent`.
3. **Decide.** If $i$ is the last model ($i = n - 1$), accept its answer and stop. Otherwise, accept its answer and stop if $c \ge$ `thresholds[i]`. If $c <$ `thresholds[i]`, reject the answer and continue to model $i + 1$.

Every model is called at most once, in order. No model is called after an answer is accepted.

**Return value.** Return a dictionary with exactly these keys:

| Key | Value |
|---|---|
| `"answer"` | `content` of the last model that was called, or `None` if no model was called |
| `"model"` | `name` of that model, or `None` if no model was called |
| `"confidence"` | confidence $c$ (a float) of that reply, or `None` if no model was called |
| `"cost"` | `spent`, the sum of costs of all models that were called |
| `"accepted"` | `True` if an answer was accepted in step 3, or `False` if the budget stopped the cascade |

When the budget stops the cascade early, return the answer from the last model that was actually called, regardless of earlier models' confidences. Confidence scores from different models are not directly comparable, and later models in the list are stronger.

## Example

Suppose we have three models:

```python
small   # cost 1,  replies "Lyon"  with logprobs [-0.1, -0.5, -0.6]
medium  # cost 5,  replies "Paris" with logprobs [-0.25, -0.75]
large   # cost 20, replies "Paris" with logprobs [-0.01, -0.03]

cascade("Capital of France?", [small, medium, large], [-0.2, -0.5], budget=30)
# {"answer": "Paris", "model": "medium", "confidence": -0.5, "cost": 6, "accepted": True}
```

Step by step:

- `small`: `0 + 1 <= 30`, so it is called (`spent` becomes `1`). Its confidence is $(-0.1 - 0.5 - 0.6) / 3 = -0.4$. Since $-0.4 < -0.2$, the answer is rejected.
- `medium`: `1 + 5 <= 30`, so it is called (`spent` becomes `6`). Its confidence is $(-0.25 - 0.75) / 2 = -0.5$. Since $-0.5 \ge -0.5$, the answer is accepted. `large` is never called, and the total cost is `6` instead of `20`.

With `budget=5`, calling `medium` would raise `spent` to `6`, which exceeds the budget. The cascade stops before calling `medium` and returns the reply from `small`:

```python
cascade("Capital of France?", [small, medium, large], [-0.2, -0.5], budget=5)
# {"answer": "Lyon", "model": "small", "confidence": -0.4, "cost": 1, "accepted": False}
```

Notice why the mean matters instead of the sum. The sum of log-probabilities is $-1.2$ for the three-token reply and $-1.0$ for the two-token reply. Summing penalizes every extra token, which would push long answers toward unnecessary escalation.

## Constraints

- $1 \le n \le 8$, `len(thresholds) == n - 1`, every threshold is $\le 0$, and `budget >= 0`.
- Costs and budgets are integers, so `cost` is compared exactly.
- `answer`, `model`, and `accepted` are compared exactly. `confidence` is compared with an absolute tolerance of `1e-9`.
- Whenever a confidence equals its threshold, the token log-probabilities are binary fractions so the mean is exact regardless of summation order. Otherwise, confidences are at least `1e-6` away from their thresholds.
- **Call accounting.** Tests inspect each model's call count and the messages it received. The set of models called, the number of calls (at most one each), and the total spend must match the specification exactly. Calling a model just to compare its output still incurs its cost.

## Why this matters

A frontier model can cost ten to a hundred times more per request than a small model. For a large share of production traffic, such as classification, extraction, and short factual lookups, the small model produces the same answer. A cascade captures that savings by sending every request to the cheap model first and escalating only uncertain replies.

The three failure modes tested here show up regularly in production routers. Using an unnormalized log-probability sum systematically escalates long answers, which are the most expensive ones to re-run. Calling the large model speculatively "to compare" pays the large model's bill on every request. Checking the budget after a call instead of before it allows requests to exceed their spending cap.
