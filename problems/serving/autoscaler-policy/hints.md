Split the policy into two independent steps. First, ignore history and compute how many replicas the current load requires. Second, use `state` to limit how far the replica count is allowed to drop based on recent recommendations.

---

For the first step, keep the arithmetic in integers for `load`, `capacity = current * cfg.target`, and the ceiling of `load / cfg.target`. Remember that `current` includes both `ready` and `starting` replicas. For the second step, store the recent `(time, recommendation)` pairs in `state`. Build a new list on every call containing only entries younger than `window_s`, then append `(metrics.now, raw)`. Scale-up returns `raw` immediately, while scale-down takes `min(current, max_in_window)`.

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
