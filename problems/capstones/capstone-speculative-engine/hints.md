Solve the stages in order, and before stage 1 decide what "where a model is" means.
Each model has a cache and a number: how many tokens of the sequence that cache
covers. If one helper feeds a model "whatever of this list you have not seen" and
another cuts its cache back to a length, what is left of the bookkeeping of
`speculative-decode-loop`? And which single token must never be in a cache at the
end of a round?

---

Wrap each model in a small object with `feed(tokens)` (call the model on
`tokens[self.length:]` with the cache, set `length = len(tokens)`, return the
logits) and `rollback(n)` (truncate if `length > n`). A round is then: compute the
limit; get `(proposer, drafts, q)` from one function; `target.feed(seq +
drafts)[-(len(drafts) + 1):]`; verify; extend `seq`; record; adapt `k`; `rollback(
len(seq) - 1)` on both. The newest token stays outside the caches because its
logits start the next round. Stage 2 changes only how a draft is picked (inverse
CDF with `u[j, 0]` instead of argmax) and the verification (the block rule with
`u[:, 1]` and `u[n, 2]`). Stage 4 puts the lookup in front of the draft loop and
returns one-hot rows as `q`, so the verification does not change at all. For stage
5 keep one draft wrapper per row and share only the target call: right-pad the
rows, read each row's verification logits at its own offset, and truncate the
shared cache to the shortest valid row.

---

The verification of stage 2, with `p` and `q` as lists of rows of Python floats
(`softmax(logits).tolist()`; one-hot rows for copied tokens) and `u` the round's
uniforms (`uniforms[r].tolist()`):

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

The batch round of stage 5, with `cached` the common cache length:

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
