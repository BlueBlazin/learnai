Implement `heal` and `healing_allowed` to perform token healing at a prompt boundary. When a prompt ends mid-token (for example, with `t` even though `true` is in the vocabulary), the model sees a boundary that rarely occurs in its training data and tends to avoid the natural continuation. Token healing trims the unstable suffix from the prompt and constrains generation to reproduce those bytes using whatever tokens the model prefers.

## Specification

```python
def heal(prompt_ids, vocab) -> tuple[list[int], bytes]: ...
def healing_allowed(required_prefix, generated_bytes, vocab) -> list[int]: ...
```

`vocab` is a [`ToyVocab`](../../../python/src/learnai/kit/vocab.py). Token ids run from `0` to `vocab.size - 1`, `vocab.bytes(t)` returns the byte string for token `t`, and special tokens (`vocab.is_special(t)`, ids 0 through 4) have the empty byte string `b""`.

Both functions must return new values without modifying their inputs.

### `heal(prompt_ids, vocab)`

Returns `(trimmed_ids, required_prefix)`, where `trimmed_ids` is a list of Python ints (the prompt with a suffix removed) and `required_prefix` is a `bytes` object (the concatenated bytes of the removed tokens that generation must reproduce).

Let $n$ be the number of tokens in `prompt_ids`. For a start index $k$ ($0 \le k < n$), let $R_k$ be the concatenated bytes of `prompt_ids[k:]`. The suffix `prompt_ids[k:]` is **healable** when both conditions hold:

1. None of the tokens in `prompt_ids[k:]` is a special token.
2. Some token $q$ in `vocab` has bytes starting with $R_k$, and the suffix is not simply that single token (`prompt_ids[k:] != [q]`).

In other words, a single vocabulary token could begin at position $k$ and cover the entire suffix. This happens either when a longer token extends a one-token suffix (`t` extended by `true`) or when one token spans multiple prompt tokens (`{` followed by `"`, covered by `{"`). A one-token suffix that no longer token extends is not healable because there is nothing to repair.

`heal` removes the **longest** healable suffix, which corresponds to the smallest healable index $k$, and returns `(list(prompt_ids[:k]), R_k)`. If no suffix is healable (including when `prompt_ids` is empty), it returns `(list(prompt_ids), b"")`.

Healability is not monotone in $k$. In the two-token prompt `t`, `r`, the suffix `r` is not healable, yet the longer suffix `t`, `r` is healable because `tr` is a token. Every candidate $k$ must be checked on its own.

### `healing_allowed(required_prefix, generated_bytes, vocab)`

Computes the allowed token ids for one generation step, returned as a list of Python ints in increasing order. `required_prefix` is the byte string returned by `heal`, and `generated_bytes` is the concatenation of the bytes of all tokens generated so far.

First, verify that `required_prefix` and `generated_bytes` agree where they overlap. If neither is a prefix of the other, raise `ValueError`.

Next, let `remaining` be the part of `required_prefix` not yet reproduced, namely `required_prefix[len(generated_bytes):]` (empty once `generated_bytes` is at least as long as `required_prefix`):

- If `remaining` is empty, the removed text has been restored and the constraint is lifted. Return **every** token id in `vocab`, special tokens included.
- Otherwise, a token `t` is allowed if and only if `vocab.bytes(t)` is non-empty and one of `vocab.bytes(t)` and `remaining` is a prefix of the other.

This rule lets the model stop short of `remaining` (`t` when `tru` remains), match it exactly, or run past it (`true` when `tru` remains, which is the purpose of healing). Special tokens are excluded while `remaining` is non-empty so the model cannot terminate or redirect generation before restoring the prompt's bytes.

## Example

Using `vocab = ToyVocab(size=64, seed=0)`, which includes `5 = b'"'`, `6 = b't'`, `7 = b'a'`, `11 = b'{"'`, `13 = b'"}'`, `15 = b'{'`, `21 = b'}'`, `27 = b'1'`, `36 = b'":'`, `46 = b'tr'`, `48 = b', '`, `49 = b'true'`, `51 = b'['`, `54 = b'tru'`, and `57 = b'": '`:

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

In the second example, the last token `"` is healable on its own because `":` extends it, and the two-token suffix `{`, `"` is also healable because `{"` is a single token. `heal` picks the longer suffix so the model can emit `{"` as one token. The three-token suffix `, `, `{`, `"` is not healable because no token starts with `, {"`. Once the model generates `{`, one byte (`"`) remains, so the four tokens starting with `"` are allowed and `{"` is no longer eligible.

## Constraints

- `vocab.size` is between 64 and 282, and prompts have at most a few hundred tokens.
- `generated_bytes` is always the concatenation of whole tokens and may be longer than `required_prefix`.
- Outputs are compared for exact equality (same token ids in the same order, same bytes).
- This backup rule is one specific design choice. Guidance, where token healing originated, describes backing up "one or more tokens", while other implementations remove exactly one. The tests follow the specification above.

## Why this matters

Prompts are rarely tokenized as a single string. Chat templates, retrieved documents, and partial structured prefixes such as `{"name": "` are often encoded separately and concatenated, leaving token boundaries that the tokenizer would never produce on the combined text. Constrained generation amplifies this problem because forced text ends wherever the grammar stops rather than on a natural token boundary. The model then continues from an out-of-distribution sequence and avoids the expected completion. In the classic example, a prompt ending in `http:` prevents a model whose vocabulary contains `://` from generating `//`.

Token healing trades a few regenerated bytes for a clean token boundary. Getting the constraint wrong breaks generation in subtle ways. Restricting the first step to the removed token itself forbids the longer tokens that justified backing up, while failing to lift the constraint once the bytes are restored leaves the model stuck unable to continue.
