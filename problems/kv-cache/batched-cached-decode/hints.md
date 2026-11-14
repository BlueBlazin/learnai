One call now returns logits of shape `(B, T, vocab_size)`: the next token of every
row is the argmax over the *last* axis of `logits[:, -1, :]`, which gives `B` tokens
at once. The single-sequence loop had one question per step ("am I done?"). What does
each row need to remember so that the batch can answer two different questions: "may
this row still collect tokens?" and "may the loop end?"

---

Keep a `finished` flag per row, next to the per-row list of tokens. After each
argmax, visit the rows: a finished row is skipped entirely (its argmax is noise); a
running row appends its token and becomes finished if that token is `eos_id` or if it
now has `max_new_tokens` tokens. The flag only ever goes from false to true. The loop
ends when *all* flags are true, and that check comes before the next model call.

---

The shape of the loop:

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

Because the "all finished" check sits between choosing and feeding, the last token of
the longest row is never fed: $g_{\max}$ calls in total.
