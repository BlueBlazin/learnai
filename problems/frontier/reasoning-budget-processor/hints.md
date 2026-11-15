The function is called with one token at a time and no history. What is the least
you need to remember between calls to know which tokens may come next? Try to name
the situations a generation can be in, and what distinguishes two calls in the same
situation (how many thinking tokens so far, how far into the end sequence).

---

Use a state `(phase, n)` with four phases: before the first token, thinking with
`n` tokens used, forcing with `n` tokens of the end sequence emitted, and answer.
Each call does two things in this order: move the state forward by `token`, then
build the row for the phase you are now in. Check "is the budget used up?" after
moving, as a separate step, so that it also fires right after the start token when
the budget is 0.

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

`where` leaves a kept logit as it is, which is what "forcing" means here: one entry
survives with its own value, however low.
