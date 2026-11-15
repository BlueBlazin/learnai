Why is a prompt that ends with the token `t` a problem when the vocabulary also has
`true`, and why is a prompt that ends with `}` not one? Phrase the difference as a
question about the vocabulary: which tokens could have started where this one starts?
Then ask the same question about the last *two* tokens taken together.

---

For `heal`, walk backwards over the prompt, growing a byte string `tail` by one token
at a time, and stop at a special token (or when `tail` is longer than the longest
token). After each step ask: does some token's byte string start with `tail`? For a
tail of one token, the token itself does not count. Remember the *last* position at
which the answer was yes; do not stop at the first no. For `healing_allowed`,
compute what is still missing, `required_prefix[len(generated_bytes):]`. Nothing
missing: all ids. Otherwise keep the tokens that agree with the missing bytes for as
long as both last.

---

"Agree for as long as both last" is one line:

```python
piece and (remaining.startswith(piece) or piece.startswith(remaining))
```

The first half lets the model rebuild the text from short tokens, the second half
lets it run past the text with a longer token, and `piece and` keeps out the special
tokens, whose empty byte string is a prefix of everything. The same trap sits in
`heal`: every token "starts with" the empty bytes of a special token, so test for
specials before comparing. Raise `ValueError` first when
`required_prefix[:m] != generated_bytes[:m]` for `m = min` of the two lengths.
