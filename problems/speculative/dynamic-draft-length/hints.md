Three small functions, each with one trap. For the confidence: the softmax
probability of the largest logit depends only on differences between logits. What
happens to `math.exp(1000.0)`, and how do you avoid ever computing it? For the
simulation: write the loop for a single round first, and decide, for each of the
five steps, which counter it touches.

---

`should_stop_drafting`: subtract the largest logit from every logit before
exponentiating. The largest becomes $e^0 = 1$, so the confidence is
`1 / sum(exp(x - top))`, and no exponent is positive.

`next_k`: compare `n_accepted` with `n_drafted`, never with `k`. A draft that was
cut to three tokens by the confidence rule and fully accepted is a success.

`simulate`: the draft loop is "count the token, then ask whether to stop", in that
order, so the low-confidence token is paid for and offered to the target. Cap the
draft at `T - pos - 1`: the last token always comes from the target.

---

The whole round:

```python
limit = min(k, total - pos - 1)
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

Record `k` in `ks` at the top of the round, before any of this.
