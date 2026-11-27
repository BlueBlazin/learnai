Solve the stages in order. Before starting stage 1, think about how to track each model's position in the sequence. Each model has a cache and an integer recording how many sequence tokens that cache covers. If one helper feeds only the unseen suffix of a token list and another truncates the cache back to a target length, most of the cache bookkeeping from `speculative-decode-loop` disappears. Also ask yourself: which single verified token must stay outside both caches at the end of a round?

---

Wrap each model in a small class with `feed(tokens)` (call the model on `tokens[self.length:]` with the current cache, set `length = len(tokens)`, and return the logits) and `rollback(n)` (truncate the cache if `length > n`).

Each round then follows a clean sequence:
1. Compute `limit = min(k, remaining - 1)`.
2. Call one proposer helper to get `(proposer, drafts, q)`.
3. Run `target.feed(seq + drafts)[-(len(drafts) + 1):]` and verify the drafts.
4. Append the accepted drafts and bonus/correction token to `seq`, record `(proposer, len(drafts), n)`, and update `k` if `adaptive` is true.
5. Call `rollback(len(seq) - 1)` on both wrappers so the newest token stays outside the caches to start the next round.

Stage 2 only changes draft selection (inverse CDF with `u[j, 0]` instead of argmax) and verification (the block rule with `u[:, 1]` and `u[n, 2]`). Stage 4 tries prompt lookup before the draft loop and returns one-hot rows for `q`, so verification stays unchanged. Stage 5 keeps one draft wrapper per row and shares only the target call. Right-pad the active rows, slice each row's verification logits at its own offset, and truncate the shared cache to the shortest valid row.

---

Stage 2 verification, where `p` and `q` are lists of float lists (`softmax(logits).tolist()`, or one-hot rows for copied tokens) and `u` is `uniforms[r].tolist()`:

```python
n = 0
while n < len(drafts) and u[n][1] * q[n][drafts[n]] < p[n][drafts[n]]:
    n += 1
if n < len(drafts):
    weights = [max(0.0, a - b) for a, b in zip(p[n], q[n])]
else:
    weights = p[n]
cdf = list(accumulate(weights))
token = sum(c <= u[n][2] * cdf[-1] for c in cdf)
```

Stage 5 batch round, where `cached` is the common cached prefix length across active rows:

```python
rows = [seqs[b][cached:] + drafts[b] for b in active]
width = max(len(row) for row in rows)
logits, cache = target([row + [0] * (width - len(row)) for row in rows], cache)
for i, b in enumerate(active):
    first = len(seqs[b]) - cached - 1  # the row that scores this row's first draft
    n, token = verify_greedy(logits[i, first : first + len(drafts[b]) + 1], drafts[b])
    valid[b] = len(seqs[b]) + n
    seqs[b] += drafts[b][:n] + [token]
# drop finished rows with cache.select(keep), then
cached = min(valid[b] for b in still_active)
cache = cache.truncate(cached)
```
