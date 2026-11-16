A prompt that ends in the middle of what the model would have written as one token
puts the model in a situation it has rarely seen. If the prompt ends with the token
`t`, the model learns something false from it: had `true` been coming, the tokenizer
would have produced `true`, not `t`, so whatever follows a lone `t` is probably not
`rue`. Token healing repairs the boundary: take the unstable tail off the prompt, and
make the model regenerate that text itself, in whatever tokens it likes. Implement
the two halves: deciding how much of the prompt to take back, and the constraint
that makes generation reproduce it.

## Specification

```python
def heal(prompt_ids, vocab) -> tuple[list[int], bytes]: ...
def healing_allowed(required_prefix, generated_bytes, vocab) -> list[int]: ...
```

`vocab` is a [`ToyVocab`](../../../python/src/learnai/kit/vocab.py): token ids are
`0` to `vocab.size - 1`, `vocab.bytes(t)` is the byte string of token `t`, and
special tokens (`vocab.is_special(t)`, ids 0 to 4) have the empty byte string `b""`.

### `heal(prompt_ids, vocab)`

Returns `(trimmed_ids, required_prefix)`: the prompt without a suffix of its tokens,
and the bytes of the removed tokens, which generation must reproduce.

Let the prompt have $n$ tokens and write $R_k$ for the concatenated bytes of
`prompt_ids[k:]`. The suffix that starts at position $k$ ($0 \le k < n$) is
**healable** when both hold:

1. none of the tokens `prompt_ids[k:]` is a special token, and
2. some token $q$ of the vocabulary has bytes that start with $R_k$, and the suffix
   is not just that token: `prompt_ids[k:] != [q]`.

In words: a single token could begin where the suffix begins and cover all of it,
either because one longer token extends it (`t` and `true`) or because one token
spells what the prompt spells with several (`{` then `"`, and the token `{"`).
A suffix of one token that no longer token extends is not healable: there is nothing
to repair.

`heal` removes the **longest** healable suffix, that is, the one with the smallest
$k$. It returns `prompt_ids[:k]` and $R_k$. If no suffix is healable (in particular
for an empty prompt) it returns the prompt unchanged and `b""`.

Healability is not monotone. In the prompt `t`, `r` (two tokens) the suffix `r` is
not healable, yet the suffix `t`, `r` is, since `tr` is a token. Every $k$ has to be
judged on its own.

### `healing_allowed(required_prefix, generated_bytes, vocab)`

The constraint for one generation step. `required_prefix` is the second result of
`heal`, and `generated_bytes` is the concatenation of the bytes of all tokens
generated so far. It returns the ids, in increasing order, of the tokens that may be
generated next.

Let `remaining` be the part of `required_prefix` that `generated_bytes` has not
reproduced yet: `required_prefix[len(generated_bytes):]`, which is empty when
`generated_bytes` is at least as long.

- If `remaining` is empty, the removed text is back and the constraint is lifted:
  **every** token id is allowed, special tokens included.
- Otherwise a token `t` is allowed exactly when `vocab.bytes(t)` is not empty and one
  of `vocab.bytes(t)` and `remaining` is a prefix of the other. The token may stop
  short of the remaining text (`t` when `tru` remains), match it, or run past it
  (`true` when `tru` remains, which is the point of healing). Special tokens are not
  allowed: they would end or redirect the output before the prompt's own text is
  restored.

The two byte strings must agree where they overlap. If neither `required_prefix`
nor `generated_bytes` is a prefix of the other, raise `ValueError`.

Both functions return new values and modify nothing. `trimmed_ids` is a list of
Python ints, `required_prefix` is `bytes`.

## Example

`vocab = ToyVocab(size=64, seed=0)`, with the tokens `5 = b'"'`, `6 = b't'`,
`7 = b'a'`, `11 = b'{"'`, `13 = b'"}'`, `15 = b'{'`, `27 = b'1'`, `36 = b'":'`,
`46 = b'tr'`, `48 = b', '`, `49 = b'true'`, `51 = b'['`, `54 = b'tru'`,
`57 = b'": '`.

```python
# '{"a": t' : the prompt ends with a token that longer tokens extend
heal([11, 7, 57, 6], vocab)  # ([11, 7, 57], b't')
healing_allowed(b"t", b"", vocab)  # [6, 46, 49, 54]     t, tr, true, tru
healing_allowed(b"t", b"tr", vocab)  # [0, 1, ..., 63]     lifted after one token

# '[1, ' then '{' then '"', tokenized separately and concatenated
heal([51, 27, 48, 15, 5], vocab)  # ([51, 27, 48], b'{"')
healing_allowed(b'{"', b"", vocab)  # [11, 15]            {" or {
healing_allowed(b'{"', b"{", vocab)  # [5, 13, 36, 57]     " or "} or ": or ":␣

# '{"a": 1}' : no token starts with '}' except '}' itself
heal([11, 7, 57, 27, 21], vocab)  # ([11, 7, 57, 27, 21], b'')
healing_allowed(b"", b"", vocab)  # [0, 1, ..., 63]
```

In the second prompt the last token `"` alone is healable (`":` extends it), and so
is the suffix `{`, `"`, because `{"` is a token. The longer suffix wins, and the
model gets the chance to write `{"` as the single token its tokenizer would have
produced. The suffix `, `, `{`, `"` is not healable: no token starts with `, {"`.
After the model has generated `{`, one byte remains, `"`, and four tokens agree with
it; `{"` is no longer among them.

## Constraints

- `vocab.size` is between 64 and 282; prompts have at most a few hundred tokens.
- `generated_bytes` is always the bytes of whole tokens, and it may be longer than
  `required_prefix`.
- Results are compared exactly: the same ids in the same order, the same bytes.
- This problem's backup rule is one precise choice. Guidance, where token healing
  comes from, describes backing up "one or more tokens"; other implementations
  remove exactly one. The tests follow the rule above.

## Why this matters

Prompts are rarely tokenized in one piece. A chat template, a retrieved document,
and a partial answer such as `{"name": "` are encoded separately and concatenated,
and every seam can leave a token boundary where the tokenizer would never have put
one. Constrained generation makes it worse, because the forced text ends wherever
the grammar says, not where a token ends. The model then continues from a sequence
that is rare in its training data and tends to avoid exactly the continuation the
prompt was written to invite: the classic example is a prompt ending in `http:`,
after which a model that has the token `://` will not produce `//`. Healing costs a
few regenerated bytes. Getting it wrong costs more: a constraint that insists on
the removed token itself forbids the longer token that was the reason to back up,
and one that never lifts leaves the model unable to write anything else.
