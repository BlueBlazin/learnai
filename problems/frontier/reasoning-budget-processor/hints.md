`budget_step` sees one token at a time and receives no token history. Ask yourself what minimal state you must carry between calls to decide which tokens are allowed next. Start by naming the distinct phases a generation can be in, then note what counter distinguishes two steps inside the same phase (how many thinking tokens have been used, or how many forced tokens have been emitted).

---

Store the state as `(phase, n)` with four phases: `"start"` before the first token, `"thinking"` with `n` thinking tokens used, `"forcing"` with `n` tokens of `forced` emitted, and `"answer"`. On every call, first advance `(phase, n)` using `token`, and then mask `logits` for the new phase. Check whether the thinking budget is exhausted right after the transition as its own step so it also triggers immediately after `start_id` when `budget == 0`.

---

```
if state is None: phase, n = "start", 0
else:
    start:    phase = "thinking" if token == start_id else "answer"
    thinking: phase = "answer" if token == end_id else (n += 1)
    forcing:  n += 1; if n == len(forced): phase = "answer"
    if phase == "thinking" and n == budget: phase, n = "forcing", 0

forcing:                 keep = (arange(V) == forced[n])
answer, with a mask:     keep = answer_mask
start, with a mask:      keep = answer_mask | (arange(V) == start_id)
otherwise:               return the raw logits
return where(keep, logits, -inf)
```

Using `where` keeps each allowed logit at its original value. Forcing a token means masking every other entry to `-inf` while leaving the forced token's own logit untouched, no matter how low it is.
