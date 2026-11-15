There are two separate questions, and it helps to answer them in two separate
steps. First: ignoring the past, how many replicas does the present load call for?
Second: given what was recommended recently, how far is the count allowed to drop?
Only the second needs `state`.

---

For the first step, compute everything in integers: the load, the capacity
`current * target`, and the ceiling of `load / target`. Remember what `current`
contains. For the second step, `state` can simply be the list of `(time,
recommendation)` pairs that are still in the window. Build a new list each time:
keep the entries that are younger than `window_s`, append the new one. Scaling up
ignores the list; scaling down takes its maximum.

---

```python
current = metrics.ready + metrics.starting
load = metrics.queue_depth + metrics.running
capacity = current * cfg.target
if abs(load - capacity) <= cfg.tolerance * capacity:
    raw = current
else:
    raw = -(-load // cfg.target)
raw = max(cfg.min_replicas, min(cfg.max_replicas, raw))
history = [(t, n) for t, n in (state or ()) if metrics.now - t < cfg.window_s]
history.append((metrics.now, raw))
if raw >= current:
    return raw, history
return min(current, max(n for _, n in history)), history
```
