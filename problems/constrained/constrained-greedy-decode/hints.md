The unconstrained loop of `cached-greedy-decode` has one line that picks a token.
The constraint only needs to change what that line can pick, and it has to be told
what was picked. Which of the constraint's four methods belongs before the argmax,
and which after it? And where does end-of-sequence fit, given that `allowed` never
contains it?

---

Keep a constraint state next to the cache; both describe "everything generated so
far" and both must move forward together. At each step build the permitted list:
`constraint.allowed(state)`, plus `eos_id` if `constraint.is_accepting(state)`. If
the list is empty, stop. Otherwise make a copy of the logits row in which every
other token is `-inf` (or look only at the permitted entries) and take the argmax.
After choosing a token that is not EOS, call `constraint.advance`.

---

To meet the cost limits, ask the constraint *before* calling the model, and feed a
token only at the start of the next step:

```
state, cache, pending = constraint.start(), None, prompt
while fewer than max_new_tokens generated:
    permitted = allowed(state) (+ [eos_id] if is_accepting(state))
    if permitted is empty: stop
    logits, cache = lm(pending, cache)
    token = argmax of logits[-1] over the permitted ids
    append token; if token == eos_id: stop
    state = constraint.advance(state, token)
    pending = [token]
```

`lm(prompt, None)` is the same as `lm(prompt)`. Because the forward call sits after
the dead-end check and the token is fed lazily, the final token is never fed and a
dead end costs nothing. Do not call `advance` with `eos_id`: it is not one of the
constraint's tokens.
