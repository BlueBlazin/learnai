Before starting the loop, compute three helper quantities: the total size of any module (the sum of its leaves), the reserve (the size of the largest indivisible module anywhere in the model), and which module owns each tied leaf. The main loop itself only ever inspects the front module of a queue and the current device.

---

Structure the loop around three outcomes for the front module:
1. It fits on the current device (or the current device is `"disk"`), so assign it and its followers and add their combined size to `used`.
2. It does not fit and is divisible, so replace it at the front of the queue with its children and stay on the current device.
3. It does not fit and is indivisible, so advance to the next device with `used = 0` and retry the same module.

Never revisit a closed device, and never skip ahead in the queue for a smaller module.

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

`followers(m)` are all keys `t` of `tied` whose owner `tied[t]` is `m` or a leaf inside `m`.
