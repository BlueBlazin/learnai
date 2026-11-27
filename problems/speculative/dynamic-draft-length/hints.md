For `should_stop_drafting`, the softmax probability of the largest logit depends only on differences between logits. Think about what happens when calling `math.exp(1000.0)` and how to avoid overflowing. For `next_k`, pay close attention to which count defines a full acceptance. For `simulate`, write out a single round first and track which counters each of the five steps updates.

---

In `should_stop_drafting`, subtract the largest logit from every entry before exponentiating. The maximum entry becomes $e^0 = 1$, so the confidence is `1.0 / sum(math.exp(x - top) for x in logits)` and no exponent is positive.

In `next_k`, compare `n_accepted` against `n_drafted` rather than `k`. A draft shortened to three tokens by confidence stopping that gets all three accepted is still a full acceptance.

In `simulate`, count each drafted token *before* checking `should_stop_drafting` so the triggering token is included in `n_drafted`, paid for, and verified. Cap the draft length at `T - pos - 1` because the last token always comes from the target model.

---

Here is the structure of one round inside `simulate`:

```python
limit = min(k, len(trace) - pos - 1)
n_drafted = 0
while n_drafted < limit:
    logits = trace[pos + n_drafted][1]
    n_drafted += 1
    if should_stop_drafting(logits, threshold):
        break
n_accepted = 0
while n_accepted < n_drafted and trace[pos + n_accepted][0]:
    n_accepted += 1
cost += n_drafted * c_draft + c_target
pos += n_accepted + 1
if schedule == "heuristic" and n_drafted > 0:
    k = next_k(k, n_accepted, n_drafted)
```

Append the current `k` to `ks` at the start of the round, before updating `k`.
