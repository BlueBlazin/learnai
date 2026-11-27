In `cached-greedy-decode`, a single line picks the next token from the logits. Adding a constraint restricts which tokens that line may choose and notifies the constraint of each choice. Which of the constraint's four methods must run before `argmax`, and which one runs after? Where does `eos_id` fit, given that `allowed` never returns it?

---

Track the constraint `state` alongside the KV `cache`, since both represent the exact sequence generated so far. On each step, build the permitted token list from `constraint.allowed(state)`, appending `eos_id` if `constraint.is_accepting(state)` is `True`. If the permitted list is empty, break out of the loop. Otherwise, copy the next-token logits row with every non-permitted token set to `-inf` (or inspect only the permitted entries) and take the `argmax`. If the chosen token is not `eos_id`, update `state = constraint.advance(state, token)`.

---

To satisfy both cost limits, check the constraint *before* calling `lm`, and pass each generated token lazily at the start of the next iteration:

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

Calling `lm(prompt, None)` is equivalent to `lm(prompt)`. Placing the forward pass after the dead-end check and feeding tokens lazily guarantees that the final token is never fed to `lm` and a dead end costs zero calls. Remember to stop before calling `advance` when `token == eos_id`, because `eos_id` is not a token of the constraint.
