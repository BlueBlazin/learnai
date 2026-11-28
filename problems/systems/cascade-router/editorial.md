## The idea

A model cascade is a sequence of models ordered by price, with a confidence gate after every model except the last. To decide whether to accept a reply, the gate uses a signal that the model already produces for free: the log-probability assigned to each generated token. When a model is uncertain among several continuations, its chosen tokens have low probability. When it is confident, token probabilities stay close to 1 (log-probabilities close to 0).

How you aggregate token log-probabilities is critical. For a reply with $T$ token log-probabilities $\ell_1, \dots, \ell_T$, their **sum**, $\sum_{t=1}^{T} \ell_t$, is the joint log-probability of the entire sequence and decreases with every token added. A clean 40-token answer at $-0.1$ per token sums to $-4.0$, which looks worse than a one-token guess at $-1.0$. Thresholding the sum ends up thresholding length. Taking the **mean**, $\frac{1}{T}\sum_{t=1}^{T} \ell_t$, computes the log of the geometric mean of token probabilities (equivalently, the negative log-perplexity of the reply). Because it is a per-token average, you can compare replies of different lengths against a single threshold.

The budget check belongs before the model call, not after. Because each model's cost is known up front, checking `spent + model.cost > budget` before calling prevents overspending. Checking after the call only detects an overrun that already happened.

When the budget stops the cascade early, returning the reply from the last model that was called is the safest choice. Confidence scores are not calibrated across different models (a small model can be confidently wrong), so ordering by model strength is more reliable than picking the highest raw confidence.

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

- **Summing log-probabilities instead of averaging them.** Short replies pass and long replies get escalated regardless of quality. In the worked example, the three-token reply sums to $-1.2$ and the two-token reply sums to $-1.0$, whereas their means are $-0.4$ and $-0.5$.
- **Averaging probabilities instead of log-probabilities.** Computing `log(mean(exp(l)))` uses the arithmetic mean of token probabilities, allowing one high-probability token to mask a low-probability guess. For `[-0.01, -3.0]`, the mean log-probability is $-1.5$, while the log of the arithmetic mean is about $-0.65$.
- **Calling all models up front.** Querying every model and then picking the first confident reply returns the same answer while paying for the entire list. A cascade saves money only by skipping later calls.
- **Checking the budget after the call.** By the time the check fails, the model has already been called and billed.
- **Using `>=` for the budget check or `>` for the threshold check.** A call that brings total spend to exactly `budget` is allowed, and a confidence equal to `thresholds[i]` is accepted. Thresholds are typically tuned on a validation set, so the boundary value itself should pass.
- **Returning the highest-confidence reply on budget exhaustion.** This favors a small, overconfident model over a stronger model that was already paid for.
- **Applying a threshold to the last model.** `thresholds` has length $n - 1$, so indexing `thresholds[i]` on the last model raises an `IndexError` or mistakenly marks the final model's reply as unaccepted.

## Complexity

A request makes at most $n$ model calls (where $n$ is the number of models) and performs $O(T)$ arithmetic for a reply of $T$ tokens. In practice, the metric of interest is expected spend. Writing $c_i$ for the cost of model $i$, expected spend is $\sum_{i=0}^{n-1} c_i \cdot P(\text{model } i \text{ is called})$. Tuning the thresholds trades this expected cost against answer quality.

## In the wild

- Model cascades are one of the three cost-reduction strategies in *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance* (arXiv 2305.05176). FrugalGPT queries a sequence of model APIs, scores each reply, and stops once a reply clears that stage's threshold. Two practical differences stand out: FrugalGPT uses a learned scoring model (a DistilBERT regressor trained on question-answer pairs) rather than token log-probabilities, and it optimizes for an *average* budget across queries rather than a hard per-request cap.
- Most OpenAI-compatible chat completion APIs return token `logprobs` on request, making length-normalized log-probability a practical default confidence signal. Because log-probability calibration varies across model families and sizes, each stage in a cascade needs its own threshold.
