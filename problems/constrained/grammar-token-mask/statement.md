A regular expression cannot count brackets, so structured output for nested formats
is specified with a context-free grammar. The decoding loop still asks the same
question at every step, "which tokens may come next?", but the answer now depends on
an unbounded amount of context: how many lists are open, which rule is in progress,
and, when the grammar is ambiguous, several readings of the text at once. Build the
matcher that answers it, token by token, for any grammar the kit can parse.

## Specification

```python
class GrammarMatcher:
    def __init__(self, grammar, vocab): ...
    def start(self): ...  # the state before any token
    def allowed(self, state) -> list[int]: ...
    def advance(self, state, token): ...  # the state after `token`
    def is_accepting(self, state) -> bool: ...
```

This is the `TokenConstraint` protocol of `constrained-greedy-decode`, for a grammar.

**The grammar.** `grammar` is a [`Grammar`](../../../python/src/learnai/kit/grammar.py),
parsed from GBNF, the notation of llama.cpp. Its language is the set of strings that
the rule `root` matches. Read the grammar from `grammar.rules`:

```python
grammar.rules  # {"root": (alternative, ...), "value": (...), ...}
```

Each rule is a tuple of alternatives, each alternative a tuple of symbols (possibly
empty), and each symbol one of

- `Chars`: exactly one character out of a set. `symbol.matches(code_point)` tests one
  character (an `int` or a one-character `str`), and `symbol.overlaps(low, high)`
  says whether the set contains any code point in the inclusive range.
- `Ref`: another rule, `symbol.name`.

Everything else in the notation has been rewritten into this shape: a literal is one
`Chars` per character, and groups and the repetitions `* + ? {m,n}` are generated
rules (`x*` is `r ::= x r | (empty)`). Two precomputed sets are yours to use:
`grammar.nullable`, the names of the rules that can match the empty string, and
`grammar.left_recursive`, the names of the rules that can begin with themselves.
Every rule can finish: there is no rule from which no string can be derived.

**The vocabulary.** `vocab` is a [`ToyVocab`](../../../python/src/learnai/kit/vocab.py):
ids `0` to `vocab.size - 1`, `vocab.bytes(t)` the byte string of token `t`. Special
tokens have the byte string `b""`. A token may be several characters, or only some
of the bytes of one character.

**The definition.** Call a byte string a *viable prefix* if some string of the
language has a UTF-8 encoding that starts with it. Let `text(state)` be the
concatenated bytes of the tokens advanced through since `start()`.

- **`allowed(state)`** returns, in increasing order, the ids of all tokens `t` such
  that `vocab.bytes(t)` is not empty and `text(state) + vocab.bytes(t)` is a viable
  prefix. Special tokens are never in the list.
- **`advance(state, token)`** returns the state for `text(state) + vocab.bytes(token)`.
  It raises `ValueError` if `token` is not allowed in `state`.
- **`is_accepting(state)`** returns `True` exactly when `text(state)` is itself the
  UTF-8 encoding of a string of the language.

States are opaque to the caller. They must be hashable, and `advance` must not
modify the state it is given: a decoding loop keeps old states to branch from.

What the definition demands of the matcher:

- **Several readings at once.** After `tr` the text may be the whole word `tr`, the
  beginning of `true`, or of `trust`. Every alternative that is consistent with the
  text stays alive until the text rules it out.
- **Rules of any shape.** Recursion on the right, in the middle, and on the left
  (`expr ::= expr "-" term | term`), directly or through other rules, and rules that
  match nothing. A matcher that eagerly expands the first symbol of a rule never
  finishes on a left-recursive one; this one must terminate and answer.
- **Tokens are not aligned with rules.** A token is allowed if *all* of its bytes can
  be read, however many rules begin and end inside it; it need not end where a rule
  ends.
- **Tokens are not aligned with characters.** The grammar is over Unicode code points
  and the vocabulary over bytes. A text that ends inside a character is viable if
  *some* character that starts with those bytes can be read next: after `"` in a
  JSON string, `b"\xe2\x82"` (the first two bytes of `€` and of 63 other characters)
  is allowed, and after `[` in a list of numbers it is not. Bytes that begin no
  character at all are never viable. The well-formed encodings are:

  | first byte | length | following bytes | code points |
  |---|---|---|---|
  | `00`–`7F` | 1 | | U+0000–U+007F |
  | `C2`–`DF` | 2 | `80`–`BF` | U+0080–U+07FF |
  | `E0`–`EF` | 3 | `80`–`BF` each | U+0800–U+FFFF, except U+D800–U+DFFF |
  | `F0`–`F4` | 4 | `80`–`BF` each | U+10000–U+10FFFF |

  The code point ranges are part of the rule: `F0 80` continues no character (its
  completions would be below U+10000), nor does `ED A0` (surrogates) or `F4 90`
  (above U+10FFFF). A text that ends inside a character is never accepting.

Write the matcher yourself. `grammar.accepts` and `grammar.accepts_prefix` are the
oracle the tests use; a solution must not call them (the harness rejects it).

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

- After `[1` the two-character token `, ` is allowed. Its comma is the literal of
  one rule and its space is the whole of another (`learnai`); the token crosses the
  boundary.
- After `[1,` the space is allowed and after `[1, ` it is not: `learnai` is at most one
  space.
- After `[1, [true]` the inner list is complete, and `root` has matched, but only
  the inner occurrence. The text is not a member of the language, so the state is
  not accepting.
- No non-ASCII token is allowed anywhere: this grammar has no such character.
- In the last state nothing is allowed and the state is accepting, so a decoding
  loop's only move is end-of-sequence.

As ids, the allowed tokens after `[1` are `[8, 12, 16, 26, 27, 32, 34, 45, 48, 50,
52, 56, 62]`.

## Constraints

- Grammars have at most a few dozen rules after rewriting. `vocab.size` is between
  64 and 282. Texts are at most a few dozen tokens long.
- Results are compared exactly with the oracle: the same ids in the same order, and
  the same accepting flag, in every state the tests reach.
- Time limit 10 s per case. One call of `allowed` on the reference takes a few
  milliseconds; nothing in the tests needs more than a straightforward
  implementation of a correct algorithm, but an algorithm that does not terminate
  on left recursion fails.
- Use `grammar.rules`, `grammar.nullable`, and `grammar.left_recursive` only. Do not
  call `grammar.accepts`, `grammar.accepts_prefix`, or the grammar's private
  methods.

## Why this matters

Grammar-constrained decoding is how engines guarantee JSON that matches a schema
with recursion, SQL in a given dialect, or a tool-call format, and this function runs
once per generated token over the whole vocabulary. Its failures are quiet. A matcher
that commits to one alternative masks out valid continuations, and the model is
pushed into whatever the surviving alternative allows. One that only accepts tokens
ending on rule boundaries starves the model of its natural multi-character tokens
and degrades the text. One that mishandles characters split across tokens either
makes non-ASCII text unreachable or lets through bytes that are not text at all, and
the response fails to decode. And one that hangs on a grammar a user wrote with
`expr ::= expr "+" term` hangs a server thread.
