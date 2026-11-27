Regular expressions cannot count nested brackets, so structured output for nested formats uses a context-free grammar. At each decoding step, the engine still asks which vocabulary tokens can legally follow the text generated so far, even when nesting is unbounded and multiple grammar alternatives remain active at once. Implement `GrammarMatcher` to answer that query token by token for any context-free grammar parsed by the kit.

![This problem is Allowed tokens and Advance state, here for a context-free grammar.](assets/map.svg)

## Specification

```python
class GrammarMatcher:
    def __init__(self, grammar, vocab): ...
    def start(self): ...  # the state before any token
    def allowed(self, state) -> list[int]: ...
    def advance(self, state, token): ...  # the state after `token`
    def is_accepting(self, state) -> bool: ...
```

This class implements the `TokenConstraint` protocol from `constrained-greedy-decode` for a context-free grammar.

**The grammar.** `grammar` is a [`Grammar`](../../../python/src/learnai/kit/grammar.py) parsed from GBNF (the notation used by llama.cpp). Its language is the set of strings matched by the `root` rule. Read the grammar from `grammar.rules`:

```python
grammar.rules  # {"root": (alternative, ...), "value": (...), ...}
```

Each rule is a tuple of alternatives, each alternative is a tuple of symbols (possibly empty), and each symbol is one of two types:

- `Chars`: matches a single Unicode character from a set. `symbol.matches(code_point)` tests one character (an `int` or one-character `str`), and `symbol.overlaps(low, high)` reports whether the set contains any code point in the inclusive range `[low, high]`.
- `Ref`: refers to another rule by `symbol.name`.

All higher-level GBNF constructs have already been rewritten into this representation. String literals become sequences of single-character `Chars` symbols, and groups or repetitions (`*`, `+`, `?`, `{m,n}`) become generated rules (for example, `x*` becomes `r ::= x r | (empty)`). Two precomputed sets are also provided: `grammar.nullable` contains the names of rules that can match the empty string, and `grammar.left_recursive` contains the names of rules that can begin with themselves. Every rule can derive at least one string.

**The vocabulary.** `vocab` is a [`ToyVocab`](../../../python/src/learnai/kit/vocab.py) with token IDs `0` through `vocab.size - 1`. `vocab.bytes(t)` returns the byte string of token `t`, and special tokens have `vocab.bytes(t) == b""`. A token may span multiple characters or only part of a single character's UTF-8 encoding.

**Definitions.** A byte string is a *viable prefix* if some string in the grammar's language has a UTF-8 encoding starting with those bytes. Let `text(state)` be the concatenation of `vocab.bytes(t)` for all tokens advanced through since `start()`.

- **`allowed(state)`** returns, in increasing order, the IDs of all tokens `t` such that `vocab.bytes(t)` is non-empty and `text(state) + vocab.bytes(t)` is a viable prefix. Special tokens are never included.
- **`advance(state, token)`** returns the new state for `text(state) + vocab.bytes(token)`. It raises `ValueError` if `token` is not allowed in `state`.
- **`is_accepting(state)`** returns `True` if and only if `text(state)` is the UTF-8 encoding of a complete string in the language.

States are opaque to the caller. They must be hashable, and `advance` must not modify the state it receives because a decoding loop may keep earlier states to branch from.

### Required behaviors

- **Multiple active readings.** After `tr`, the text might be the full word `tr`, the start of `true`, or the start of `trust`. Every consistent alternative stays alive until later bytes rule it out.
- **Arbitrary recursion and nullable rules.** Grammars may use right, middle, or left recursion (`expr ::= expr "-" term | term`), directly or through other rules, as well as rules that match the empty string. A matcher that naively expands the leftmost symbol will loop forever on left recursion. Your matcher must terminate and return the exact result.
- **Tokens crossing rule boundaries.** A token is allowed whenever all of its bytes can be consumed, regardless of how many rules open or close inside it. A token does not need to end on a rule boundary.
- **Tokens splitting UTF-8 characters.** Grammars operate on Unicode code points, while vocabularies operate on bytes. A byte string ending mid-character is viable if *some* valid Unicode character starting with those bytes can be read next. For example, after `"` in a JSON string, `b"\xe2\x82"` (the first two bytes of `€` and 63 other characters) is allowed, whereas after `[` in a list of digits it is not. Invalid UTF-8 byte sequences are never viable, and a state ending mid-character is never accepting.

  The valid UTF-8 encodings are:

  | first byte | length | following bytes | code points |
  |---|---|---|---|
  | `00`–`7F` | 1 | | U+0000–U+007F |
  | `C2`–`DF` | 2 | `80`–`BF` | U+0080–U+07FF |
  | `E0`–`EF` | 3 | `80`–`BF` each | U+0800–U+FFFF, except U+D800–U+DFFF |
  | `F0`–`F4` | 4 | `80`–`BF` each | U+10000–U+10FFFF |

  Code point bounds are strict. `F0 80` is invalid because all its completions are below U+10000 (overlong), `ED A0` is invalid because it falls in the surrogate range U+D800–U+DFFF, and `F4 90` is invalid because it exceeds U+10FFFF.

Write the matcher yourself. `grammar.accepts` and `grammar.accepts_prefix` are the test oracle and must not be called (the harness rejects solutions that call them).

## Example

```python
grammar = Grammar.parse("""
    root  ::= "[" ws (value ("," ws value)*)? "]"
    value ::= [0-9]+ | "true" | "false" | root      # nested lists
    ws    ::= " "?
""")
vocab = ToyVocab(size=64, seed=0)
matcher = GrammarMatcher(grammar, vocab)
```

Advancing through the tokens `[`, `1`, `,`, (space), `[`, `tr`, `u`, `e`, `]`, `]`:

| text so far | allowed tokens | accepting |
|---|---|---|
| (empty) | `[` | no |
| `[` | the ten digits, `t` `tr` `tru` `true`, `f` `fa` `fal` `fals` `false`, `[`, `]`, and the space | no |
| `[1` | the ten digits, `,`, `, `, and `]` | no |
| `[1,` | the digits, the pieces of `true` and `false`, `[`, and the space | no |
| `[1, ` | the same without the space | no |
| `[1, [` | as after `[` | no |
| `[1, [tr` | `u` | no |
| `[1, [tru` | `e` | no |
| `[1, [true` | `,`, `, `, `]` | no |
| `[1, [true]` | `,`, `, `, `]` | no |
| `[1, [true]]` | (none) | yes |

- After `[1`, the two-character token `, ` is allowed. Its comma comes from the literal in `root` and its space matches `ws`, so the token crosses a rule boundary.
- After `[1,`, the space is allowed, and after `[1, ` it is not, because `ws` matches at most one space.
- After `[1, [true]`, the inner list has finished matching `root`, but the outer `root` is still open. The state is not accepting.
- No non-ASCII token is allowed at any step because this grammar contains only ASCII characters.
- In the final state, no tokens are allowed and `is_accepting` is `True`, so a decoding loop must stop with end-of-sequence.

As token IDs, the allowed tokens after `[1` are `[8, 12, 16, 26, 27, 32, 34, 45, 48, 50, 52, 56, 62]`.

## Constraints

- Grammars have at most a few dozen rules after rewriting. `vocab.size` is between 64 and 282. Test sequences are at most a few dozen tokens long.
- Outputs are compared exactly with the oracle, requiring identical token ID lists in increasing order and identical `is_accepting` values at every visited state.
- Time limit is 10 s per test case. A single call to `allowed` takes a few milliseconds on the reference implementation. A straightforward correct algorithm passes easily, while an algorithm that loops on left recursion times out.
- Use only `grammar.rules`, `grammar.nullable`, and `grammar.left_recursive`. Do not call `grammar.accepts`, `grammar.accepts_prefix`, or private methods of `Grammar`.

## Why this matters

Grammar-constrained decoding lets inference engines guarantee valid recursive JSON schemas, SQL dialects, or structured tool-call formats. The matcher runs over the entire vocabulary on every generated token, and bugs in it fail silently.

A matcher that commits too early to one alternative masks out valid continuations and forces the model down the wrong branch. One that requires tokens to end on rule boundaries blocks natural multi-character tokens and degrades output quality. Mishandling multi-byte UTF-8 characters either makes non-ASCII text unreachable or allows invalid byte sequences that fail to decode. Finally, a naive stack expander that loops on a user-supplied left-recursive rule like `expr ::= expr "+" term` hangs the inference worker.
