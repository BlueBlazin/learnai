Three numbers are needed before the loop starts: the size of any module (a sum over
its leaves), the reserve (the largest module that can never be divided), and for
each tied module, where its owner is. The loop itself only ever looks at one
module, the front of a queue, and one device.

---

Write the loop with three outcomes for the front module: it fits (record it and
its followers, add to `used`), it does not fit but can be opened (put its children
at the front of the queue and stay on the device), or it does not fit and is
indivisible (move to the next device with `used = 0` and try the same module
again). `"disk"` accepts everything. Never look back at an earlier device, and
never look ahead in the queue for something smaller.

---

```python
devices = [*max_memory, "disk"]
budget = dict(max_memory)
if max_memory:
    budget[devices[0]] -= reserve  # largest leaf or no-split module
queue = [m for m in modules if m[0] not in tied]
current, used = 0, 0
while queue:
    m = queue.pop(0)
    need = size(m) + sum(size(t) for t in followers(m))
    if devices[current] == "disk" or used + need <= budget[devices[current]]:
        ...  # place m and its followers
    elif not indivisible(m):
        queue[:0] = m[2]
    else:
        current, used = current + 1, 0
        queue.insert(0, m)
```

`followers(m)` are the keys `t` of `tied` whose owner `tied[t]` is `m` or one of the
leaves inside `m`.
