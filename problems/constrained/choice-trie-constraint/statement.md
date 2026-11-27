A classifier built on an LLM must output one of a fixed set of strings, such as `positive`, `negative`, or `neutral`, and nothing else. Implement `build_trie` to store the allowed token sequences in a prefix tree (trie) and `allowed_next` to return which tokens may follow a generated prefix. The end-of-sequence (EOS) token must be allowed if and only if the generated sequence equals a complete choice, even when that choice is also a prefix of a longer choice.

![This problem is Allowed tokens and Advance state, here a walk down a trie of the choices.](assets/map.svg)

## Specification

```python
def build_trie(choices: list[list[int]], eos_id: int): ...
def allowed_next(trie, generated: list[int]) -> list[int]: ...
```

**`build_trie(choices, eos_id)`** takes a list of pre-tokenized choices, where `choices[i]` is the list of token IDs for choice `i`, along with the end-of-sequence token ID `eos_id`. It returns a trie in any representation you choose. The tests never inspect the trie directly and only pass it back to `allowed_next`.

**`allowed_next(trie, generated)`** takes that trie and the list of token IDs generated so far. It returns a list of distinct Python `int` values in **increasing order** containing:

- every token `t` such that `generated + [t]` is a prefix of at least one choice (a choice counts as a prefix of itself)
- `eos_id`, if and only if `generated` equals one of the choices

`eos_id` is sorted with the other token IDs by its numeric value and has no special position.

Additional requirements:

- When `generated` is empty, return the sorted unique first tokens of all choices.
- When `generated` is not a prefix of any choice, return `[]`.
- One choice may be a prefix of another (for example, `[7]` and `[7, 12]`). After `[7]`, both `eos_id` and `12` are allowed.
- Duplicate choices may appear in `choices`, and multiple choices may share the same next token. Each allowed token ID must appear only once in the returned list.
- There is at least one choice, every choice has at least one token, and no choice contains `eos_id`. `generated` never contains `eos_id` because decoding stops once EOS is emitted.
- `build_trie` must not modify `choices`, and `allowed_next` must not modify `trie` or `generated`. Multiple tries may exist at the same time.

## Example

Suppose the tokenizer has the tokens `7 = "yes"`, `12 = " please"`, `9 = "no"`, `4 = "t"`, and `3 = " sure"`, with EOS token `10`. The choices `yes`, `yes please`, `no`, and `not sure` are encoded as:

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

![The trie of the example. Under each node is what `allowed_next` returns there, with EOS where a choice ends.](assets/trie.svg)

At the start, EOS is not allowed because the empty output is not a choice. After `[9, 4]`, the output reads `not`, which is only a proper prefix of `not sure`, so the model must continue. After `[7]`, both stopping and continuing are valid, and the model's logits decide between them.

## Constraints

- At most 1,000 choices of at most 64 tokens each. Token IDs and `eos_id` are non-negative ints.
- Results are compared for exact equality, including order.
- There is no cost limit beyond the time limit, but the intended trie answers each query in time proportional to `len(generated)` plus the size of the answer, without scanning every choice.

## Why this matters

Picking one string from a fixed list is the simplest and most widely used structured-output feature. It powers classification, routing, multiple-choice evaluation, and enum fields in tool arguments. vLLM exposes it as the `choice` option in structured outputs, and token tries are also used in generative retrieval to restrict generation to valid entity names or document IDs.

Getting EOS handling wrong breaks the contract directly. Allowing EOS too early lets the server return a truncated prefix like `neg`, while forgetting that a shorter choice like `yes` can end forces the model to generate `yes please` every time.
