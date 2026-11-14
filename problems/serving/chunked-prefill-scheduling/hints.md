Every step has a fixed budget and two kinds of customers: sequences that need one
token each to keep streaming, and prompts that could each absorb the whole budget.
Which kind has to be paid first if the time between two tokens of a running sequence
must never exceed the length of one step?

---

Use one running total. Start with `budget = max_batch_tokens`, take the decodes
(`running[:max_batch_tokens]`) and subtract their number. Then go through
`prefilling` followed by `waiting` as one line: each candidate takes the smallest of
three numbers (what it still needs, the threshold, what is left), and you stop as soon
as nothing is left. You never need to look ahead or sort.

---

```python
for seq_id, remaining in [*prefilling, *waiting]:
    if budget == 0:
        break
    n = min(remaining, long_prefill_threshold, budget)
    prefill.append((seq_id, n))
    budget -= n
```

The `break` on an empty budget is what keeps zero-token chunks out of the list (the
engine rejects them). Nothing else is needed: do not add a sequence to `decode`
because its chunk happens to complete its prompt, and do not skip a prompt because it
does not fit whole.
