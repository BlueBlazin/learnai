Why does a prompt ending in the token `t` cause trouble when the vocabulary also contains `true`, whereas a prompt ending in `}` does not? Think about which vocabulary tokens could have started where that last token starts. Then ask the same question for the last *two* tokens taken together.

---

For `heal`, walk backward from the end of `prompt_ids`, prepending each token's bytes to a running byte string `tail`. Stop early only if you hit a special token or if `tail` becomes longer than the longest token in `vocab`. At each position `k`, check whether any token's bytes start with `tail` (for a one-token suffix, ignore the token matching itself). Record the *earliest* position `k` where this holds, and do not stop on the first position that fails.

For `healing_allowed`, first verify that `required_prefix` and `generated_bytes` agree on their shared prefix. Then compute `remaining = required_prefix[len(generated_bytes):]`. If `remaining` is empty, return all token ids. Otherwise, keep every non-special token whose bytes agree with `remaining` for as long as both last.

---

"Agree with `remaining` for as long as both last" reduces to one condition:

```python
piece and (remaining.startswith(piece) or piece.startswith(remaining))
```

The first clause lets the model rebuild the removed text from shorter tokens, the second lets it overshoot with a longer token, and `piece and` filters out special tokens whose empty byte string `b""` is a prefix of everything. Watch out for the same empty-bytes trap in `heal` by checking for special tokens before comparing prefixes. Before slicing `remaining`, let `m = min(len(required_prefix), len(generated_bytes))` and raise `ValueError` if `required_prefix[:m] != generated_bytes[:m]`.
