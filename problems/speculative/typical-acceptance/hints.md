The row alignment and prefix stopping rule match greedy verification. Row `i` corresponds to `draft_tokens[i]`, the first failure ends the accepted prefix, and one extra row supplies `next_token`. Only the per-token pass test changes from an argmax match to a per-row probability threshold. What quantity do you need from each row of `p` to compute that threshold?

---

Compute the entropy of all `K + 1` rows at once in nats with `-(p * log(p)).sum(-1)`. To prevent `0 * log(0)` from producing `NaN`, replace zeros in the argument to `log` with `1` using a `where` mask so those terms evaluate to `0 * 0`. Each row's threshold is then the smaller of `epsilon` and `delta * exp(-entropy)`.

Gather the target probability of each draft token with a single indexing operation over rows `0..K-1` and columns `draft_tokens`.

---

Given `prob` of shape `(K,)` and `threshold` of shape `(K + 1,)`:

```python
accepted = prob > threshold[:K]  # strict
n_accepted = cumprod(accepted).sum()  # 1 up to the first False, then 0
next_token = argmax(p[n_accepted])  # the target's greedy token at that row
```

Convert both `n_accepted` and `next_token` to Python `int`s before returning.
