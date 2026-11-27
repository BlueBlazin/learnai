Every step has a fixed token budget and two kinds of work. Running sequences need 1 token each to keep streaming, while prompts can each consume the entire budget. Which group must be scheduled first if the gap between two tokens of a running sequence must never exceed one step?

---

Track a single remaining budget. Start with `budget = max_batch_tokens`, take `running[:max_batch_tokens]` for `decode`, and subtract `len(decode)` from `budget`. Then iterate through `prefilling` followed by `waiting` in order. Give each candidate the smallest of its remaining tokens, `long_prefill_threshold`, and `budget`, and stop as soon as `budget` reaches 0. You never need to sort or look ahead.

---

```python
for seq_id, remaining in [*prefilling, *waiting]:
    if budget == 0:
        break
    n = min(remaining, long_prefill_threshold, budget)
    prefill.append((seq_id, n))
    budget -= n
```

Breaking when `budget == 0` keeps zero-token chunks out of `prefill`, which the engine would otherwise reject. Do not add a sequence to `decode` just because its chunk finishes its prompt, and do not skip a prompt just because it cannot fit whole.
