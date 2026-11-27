Two questions settle the structure. First, which messages may never be dropped? Second, at which indices can you cut the remaining messages so that the kept history always starts with a user message and never orphans an assistant reply or tool result?

---

Separate the system message into `head` if `messages[0]["role"] == "system"` (or `[]` otherwise), and put the remaining messages in `rest`. In `rest`, every index holding a `"user"` message is a valid cut point that drops only whole turns. The candidate lists are `head + rest[i:]` for each such index `i`, from the first (drop nothing) to the last (keep only the last turn). The prompt token budget is `max_context - max_new_tokens`.

---

Check the candidates from longest to shortest and return the first one within budget:

```python
for i in user_indices:  # ascending order drops oldest turns first
    kept = head + rest[i:]
    if count_tokens(kept) <= max_context - max_new_tokens:
        return kept
raise ValueError(...)
```

Pass each full candidate list to `count_tokens`, since token counts are not additive. Slicing and concatenating always create a new list, so `messages` is never modified.
