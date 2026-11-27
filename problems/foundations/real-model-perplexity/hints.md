A window of `T` tokens produces `T` rows of logits, but most of them predict tokens that an earlier window already scored with more context. If you track one integer between windows—the first position not yet scored—which rows of the next window correspond to new tokens?

---

Track `scored`, the first position that no window has scored yet. Start with `scored = 1`, because nothing precedes token 0. For each `begin` in `range(0, N, stride)`:

1. Set `end = min(begin + max_len, N)` and call `logits = forward(token_ids[begin:end])`.
2. The new target positions for this window are `scored` through `end - 1`.
3. Position `p` is predicted from the row for position `p - 1`, which sits at local index `p - 1 - begin` in `logits`.
4. Add those log-probabilities to a running sum, update `scored = end`, and break once `end == N`.

Divide the total by `N - 1` once at the very end, not per window.

---

In PyTorch, the loop body is:

```python
logp = torch.log_softmax(forward(token_ids[begin:end]), dim=-1)
rows = torch.arange(scored - 1 - begin, end - 1 - begin)
total = total + logp[rows, token_ids[scored:end]].sum()
```

Return `torch.exp(-total / (N - 1))`. Use `log_softmax` (or `logits - logsumexp`) rather than `log(softmax(...) + eps)`. The real model assigns some tokens probabilities far smaller than any epsilon you might pick, so an epsilon silently clamps their log-probabilities.
