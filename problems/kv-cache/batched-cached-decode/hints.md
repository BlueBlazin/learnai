Each model call now returns logits of shape `(B, T, vocab_size)`. Taking the argmax over the last axis of `logits[:, -1, :]` gives the next token for all `B` rows at once. The single-sequence loop only had to ask "am I done?" at each step. What per-row state do you need so the batch can answer both "should this row still collect tokens?" and "can the loop end?"

---

Keep a `finished` boolean flag per row alongside each row's list of generated tokens. After each argmax, inspect every row. Skip finished rows completely (their argmax is noise). For each active row, append its new token and mark it finished if that token equals `eos_id` or if the row now has `max_new_tokens` tokens. Once `finished[b]` becomes `True`, it never changes back. End the loop when *all* flags are `True`, and run that check before making the next model call.

---

The loop has this structure:

```
logits, cache = lm(prompts)                      # (B, P) in one call
loop:
    tokens = argmax(logits[:, -1, :], axis=-1)   # B tokens, as Python ints
    for each row b that is not finished:
        append tokens[b]; update finished[b]
    if all rows are finished: return the rows
    feed = [[eos_id] if finished[b] else [last token of row b] for each row b]
    logits, cache = lm(feed, cache)              # (B, 1)
```

Because the `all(finished)` check sits between selecting tokens and feeding them back, the last token of the longest row is never fed into `lm`, giving $g_{\max}$ calls in total.
