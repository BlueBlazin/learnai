Two questions settle the structure. Which messages may never be dropped? And at which
positions is it legal to cut the rest, so that what remains still starts with a
question and every answer and tool result still has its question?

---

Split off the system message if `messages[0]` is one. In the rest, every index that
holds a `user` message is a legal place to start: cutting there drops whole turns and
nothing else. So the candidates are `system + rest[i:]` for each such index `i`, from
the first (drop nothing) to the last (keep only the last turn). The budget for the
prompt is `max_context - max_new_tokens`.

---

Walk the candidates from the longest to the shortest and return the first one whose
count is within the budget:

```python
for i in user_indices:  # ascending: oldest turn first
    kept = head + rest[i:]
    if count_tokens(kept) <= max_context - max_new_tokens:
        return kept
raise ValueError(...)
```

Count each candidate as a whole list, since the count is not additive. Slicing and
concatenating build a new list every time, so the argument is never modified and the
result is never the caller's own list.
