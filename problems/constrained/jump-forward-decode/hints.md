In `constrained-greedy-decode` the permitted set is computed before the model is
called. Look at a step where that set has one element. What can the argmax over the
masked logits return, and what did the forward call contribute? And what must be
true of the model's cache before the *next* step at which it has a real choice?

---

Keep a list of tokens the model has not seen yet, starting with the whole prompt. At
each step build the permitted list (`allowed(state)`, plus `eos_id` if
`is_accepting(state)`). Empty: stop. One element: that is the next token, no call.
Two or more: call the model once with everything in the list, empty the list, and
take the best permitted token from the last row of logits. Either way, advance the
constraint and append the token to the list of unseen tokens. Count the permitted
set, not the allowed list: one allowed token plus EOS is a choice, and EOS alone is
forced.

---

```
state, cache, pending, out = constraint.start(), None, list(prompt), []
while len(out) < max_new_tokens:
    permitted = allowed(state) (+ [eos_id] if is_accepting(state))
    if not permitted: break
    if len(permitted) == 1:
        token = permitted[0]
    else:
        logits, cache = lm(pending, cache); pending = []
        token = argmax of logits[-1] over the permitted ids
    out.append(token)
    if token == eos_id: break
    state = constraint.advance(state, token)
    pending.append(token)
return out
```

`lm(pending, None)` is `lm(pending)`. The limits follow from this shape: the model is
called only in the `else` branch (once per free step), `pending` always holds exactly
the tokens since the last call, and whatever is in it when the loop ends is never
fed. Checking `len(out) < max_new_tokens` once per token, forced or free, is what
cuts a forced run at the limit.
