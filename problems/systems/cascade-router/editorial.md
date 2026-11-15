## The idea

A cascade is a chain of models ordered by price, with a gate after each one. The
gate needs a number that says how far to trust the answer, and the cheapest such
number comes with the answer: the log-probability the model assigned to each token
it emitted. A model that hesitated between several continuations emits tokens with
low probability; a model that was sure emits them near probability 1.

How the token values are combined matters. Their **sum** is the log-probability of
the whole answer, $\log p(y) = \sum_t \ell_t$, and it falls with every token: a
perfectly ordinary 40-token answer at $-0.1$ per token scores $-4$, far below a
one-word guess at $-1$. A threshold on the sum is a threshold on length. The
**mean** $\frac{1}{T}\sum_t \ell_t$ is the log of the geometric mean of the token
probabilities, a per-token quantity that can be compared across answers of
different lengths. (It is the negative of the log of the answer's perplexity.)

The budget is the other half. The price of a call is known before the call, so the
only correct place for the check is in front of it: `spent + cost > budget` means
"do not make this call". Checking after the call reports the overrun instead of
preventing it.

When the budget ends the cascade early, something still has to be returned. The
answer of the last model called is the right choice: the confidences of two
different models are not on a common scale (a small model can be confidently
wrong), while the order of the list says which model is the stronger one.

## Reference solution (Python)

```python
def cascade(prompt, models, thresholds, budget):
    messages = [{"role": "user", "content": prompt}]
    result = {"answer": None, "model": None, "confidence": None, "cost": 0, "accepted": False}

    for i, model in enumerate(models):
        if result["cost"] + model.cost > budget:
            break
        reply = model.complete(messages)
        result["cost"] += model.cost

        logprobs = reply["logprobs"]
        result["answer"] = reply["content"]
        result["model"] = model.name
        result["confidence"] = sum(logprobs) / len(logprobs)

        last = i == len(models) - 1
        if last or result["confidence"] >= thresholds[i]:
            result["accepted"] = True
            break

    return result
```

## Common mistakes

- **Confidence as the sum of the log-probabilities.** Short answers pass, long
  answers are escalated, whatever the model thought of them. In the worked example
  the three-token answer scores $-1.2$ and the two-token answer $-1.0$; with the
  mean they score $-0.4$ and $-0.5$.
- **Averaging probabilities instead of log-probabilities.** `log(mean(exp(l)))` is
  the arithmetic mean of the token probabilities. One confident token hides a
  token the model was guessing at: `[-0.01, -3.0]` has mean $-1.5$ but an
  arithmetic-mean score of about $-0.65$.
- **Calling the large model anyway.** Fetching every model's answer first and then
  picking the first confident one returns the same answer and pays for the whole
  list. The saving of a cascade is the calls it does not make.
- **Checking the budget after the call.** The call that breaks the budget has then
  been made and paid for.
- **`>=` in the budget check, or `>` in the threshold check.** A call that brings
  the spend to exactly the budget is allowed, and a confidence exactly at the
  threshold is accepted. Both are stated as inequalities for a reason: thresholds
  are tuned on a validation set and a tie is the tuned value itself.
- **Returning the most confident answer when the budget runs out.** It prefers the
  small model whenever the small model was overconfident, which is the case a
  cascade exists to correct.
- **Applying a threshold to the last model**, and returning "not accepted" (or
  indexing past the end of `thresholds`) when it fails.

## Complexity

At most $n$ model calls and $O(T)$ arithmetic per reply. The quantity that matters
is the expected spend, $\sum_i c_i \cdot P(\text{stage } i \text{ is reached})$,
which the thresholds trade against accuracy.

## In the wild

- The cascade is one of the three strategies of *FrugalGPT: How to Use Large
  Language Models While Reducing Cost and Improving Performance* (arXiv 2305.05176):
  query a list of model APIs in sequence, score each answer, return it if the score
  exceeds that stage's threshold, and otherwise query the next. Two differences
  from this problem are worth knowing. FrugalGPT's score comes from a small trained
  model (a DistilBERT regressor that reads the question and the answer), not from
  token log-probabilities; and its budget constrains the *average* cost per query
  when the list and the thresholds are learned, where this problem enforces a hard
  limit per request.
- Token log-probabilities are what an API gives you for free (`logprobs` in
  OpenAI-compatible chat completions), which is why length-normalized
  log-probability is the usual first confidence signal. It is not calibrated
  across models, so each stage needs its own threshold.
