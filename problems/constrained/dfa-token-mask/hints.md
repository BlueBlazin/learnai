The automaton speaks bytes and the model speaks tokens. For one token, in one state,
what is the smallest experiment that tells you whether the token can come next? And
what does that experiment say about a token whose byte string is empty?

---

Loop over every token id. Fetch its bytes with `vocab.bytes(token)`. If there are
none, skip the token. Otherwise feed the bytes to the automaton one after another,
starting from `state` (`dfa.walk(state, piece)` does exactly this), and look only at
whether the result is `DEAD`. Do not ask whether the end state is accepting, and do
not stop after the first byte.

---

Put the walk in one helper and build both functions on it:

```python
def walk_token(dfa, state, token, vocab):
    piece = vocab.bytes(token)
    return dfa.walk(state, piece) if piece else DEAD
```

`allowed_tokens` keeps the ids for which the helper's result is not `DEAD`, in the
order of `range(vocab.size)`, which is already increasing. `advance` returns the
helper's result, or raises `ValueError` when it is `DEAD`. Because the walk is over
bytes, a token that ends in the middle of `€` needs no special case: the automaton
simply stops in an intermediate state.
