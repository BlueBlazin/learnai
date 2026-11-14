A window of `T` tokens gives you `T` rows of logits, but most of them answer questions
that an earlier window has already answered with more context. If you remember one
number between windows, the position up to which the text is already scored, which
rows of the next window are new?

---

Keep `scored`, the first position that no window has scored yet. It starts at 1,
because nothing can predict token 0. For each `begin` in `range(0, N, stride)`:

1. `end = min(begin + max_len, N)` and `logits = forward(token_ids[begin:end])`.
2. The new targets are the positions `scored` to `end - 1`.
3. The row that predicts position `p` is `p - 1 - begin`: the model has just read
   position `p - 1`, which is row `p - 1 - begin` of this window.
4. Add their log-probabilities to a running sum, set `scored = end`, and stop when
   `end == N`.

Divide by `N - 1` at the very end, once, not per window.

---

In PyTorch the body of the loop is

```python
logp = torch.log_softmax(forward(token_ids[begin:end]), dim=-1)
rows = torch.arange(scored - 1 - begin, end - 1 - begin)
total = total + logp[rows, token_ids[scored:end]].sum()
```

and the result is `torch.exp(-total / (N - 1))`. Use `log_softmax` (or
`logits - logsumexp`), never `log(softmax(...) + eps)`: the real model assigns some
tokens a probability far below any epsilon you would pick, and the epsilon silently
replaces their log-probability.
