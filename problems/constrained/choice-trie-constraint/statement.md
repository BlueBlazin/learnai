A classifier built on an LLM must answer with one of a fixed set of strings, such as
`positive`, `negative`, or `neutral`, and with nothing else. Implement the constraint:
a prefix tree (trie) over the token sequences of the choices, and the function that
says which tokens may come next. The subtle part is the end: the end-of-sequence
token must be allowed exactly when the output is a complete choice, also when that
choice is the beginning of a longer one.

## Specification

```python
def build_trie(choices: list[list[int]], eos_id: int): ...
def allowed_next(trie, generated: list[int]) -> list[int]: ...
```

**`build_trie(choices, eos_id)`** receives the choices, already tokenized: `choices[i]`
is the list of token ids of choice `i`. It returns a trie in any representation you
like. The tests never look inside it; they only pass it back to `allowed_next`.

**`allowed_next(trie, generated)`** receives that trie and the token ids generated so
far, and returns the list of token ids that may be generated next:

- every token `t` such that `generated + [t]` is a prefix of at least one choice
  (a choice counts as a prefix of itself);
- and `eos_id`, exactly when `generated` is equal to one of the choices.

The result is a list of distinct Python ints in **increasing order**. `eos_id` is
sorted in with the other ids by its numeric value; it has no special position.

Details:

- `generated` may be empty: the answer is then the set of first tokens.
- If `generated` is not a prefix of any choice, nothing can follow: return `[]`.
- One choice may be a prefix of another (`[7]` and `[7, 12]`). After `[7]` both
  `eos_id` and `12` are allowed.
- The same choice may be listed more than once, and several choices may continue with
  the same token. The result still contains each id once.
- There is at least one choice, every choice has at least one token, and no choice
  contains `eos_id`. `generated` never contains `eos_id` either: once EOS is
  generated, decoding is over.
- `build_trie` does not modify `choices`, and `allowed_next` does not modify the trie
  or `generated`. Several tries may exist at the same time.

## Example

Suppose the tokenizer has the tokens `7 = "yes"`, `12 = " please"`, `9 = "no"`,
`4 = "t"`, `3 = " sure"`, and EOS is token `10`. The choices `yes`, `yes please`, `no`,
and `not sure` are then:

```python
trie = build_trie([[7], [7, 12], [9], [9, 4, 3]], eos_id=10)
```

```
(start) ─ 7 "yes" ● ─ 12 " please" ●
        └ 9 "no"  ● ─ 4 "t" ─ 3 " sure" ●          ● = a choice ends here
```

```python
allowed_next(trie, [])            # [7, 9]
allowed_next(trie, [7])           # [10, 12]   "yes" is complete, or " please" follows
allowed_next(trie, [7, 12])       # [10]
allowed_next(trie, [9])           # [4, 10]    EOS is sorted in by its id
allowed_next(trie, [9, 4])        # [3]        "not" is not a choice: no EOS
allowed_next(trie, [9, 4, 3])     # [10]
allowed_next(trie, [12])          # []         no choice starts with " please"
allowed_next(trie, [7, 9])        # []
```

At the start EOS is not allowed, because the empty output is not a choice. After
`[9, 4]` the output reads `not`, a proper prefix of `not sure` only, so the model must
go on. After `[7]` both ending and continuing are legal, and the model's logits decide.

## Constraints

- At most 1,000 choices of at most 64 tokens each. Token ids and `eos_id` are
  non-negative ints.
- Results are compared exactly, including order.
- There is no cost limit beyond the time limit, but the intended structure answers a
  query in time proportional to `len(generated)` plus the size of the answer, without
  looking at every choice.

## Why this matters

"Pick one of these" is the simplest structured-output feature and one of the most
used: classification, routing, multiple-choice evaluation, and enum fields in tool
arguments. vLLM exposes it as the `choice` option of its structured outputs, and the
trie over token ids is the classic way to implement it, also used to restrict
generation to valid entity names or document identifiers in generative retrieval.
The failure modes are concrete: allow EOS too early and the server returns `neg`;
forget that `yes` may also end and the model is forced into `yes please` every time.
