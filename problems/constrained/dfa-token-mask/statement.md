A regular expression describes the text you want, character by character, but a model
generates **tokens**, and a token can be several characters, half a character, or a
special marker with no text at all. Implement the bridge: given the automaton of the
regular expression and its current state, compute which tokens may be generated next,
and move the automaton forward once a token has been chosen.

## Specification

```python
def allowed_tokens(dfa, state, vocab) -> list[int]: ...
def advance(dfa, state, token, vocab) -> int: ...
```

**The automaton.** `dfa` is a [`CharDFA`](../../../python/src/learnai/kit/dfa.py), a
deterministic automaton that reads **bytes**. Its states are non-negative ints.
`dfa.step(state, byte)` returns the state after reading one byte (an int in
`[0, 256)`), or the constant `DEAD` (`-1`) if no string of the language continues that
way. `dfa.walk(state, data)` does the same for a `bytes` object and stops at the first
byte that kills the walk. `dfa.is_accepting(state)` says whether the bytes read so far
form a complete string of the language. The automaton is *trimmed*: from every state
some continuation reaches an accepting state, so a walk that is not `DEAD` has read a
prefix of a string of the language.

**The vocabulary.** `vocab` is a [`ToyVocab`](../../../python/src/learnai/kit/vocab.py).
Token ids are `0` to `vocab.size - 1`, and `vocab.bytes(token)` is the byte string the
token stands for. Three kinds of tokens matter here:

- a token may have several bytes (`b'true'`, `b'": '`), which take the automaton
  through several transitions;
- a token may be a fragment of a UTF-8 character (`b'\xe2\x82'` is the first two of
  the three bytes of `€`);
- special tokens such as end-of-sequence have the **empty** byte string `b""`.

**`allowed_tokens(dfa, state, vocab)`** returns the ids, in increasing order, of all
tokens `t` such that

1. `vocab.bytes(t)` has at least one byte, and
2. reading all of its bytes, in order, starting from `state`, never yields `DEAD`.

Nothing else is required. In particular the state after the token does **not** have
to be accepting: later tokens can complete the string. A token is rejected as a whole
as soon as one of its bytes dies, however many of its bytes were fine before that.

Special tokens fail condition 1 and are never in the list, in any state. Whether
generation may *end* is a separate question that the decoding loop answers with
`dfa.is_accepting(state)`; it is not part of this function.

**`advance(dfa, state, token, vocab)`** returns the state reached by reading the bytes
of `token` from `state`. If `token` is not allowed in `state` (it would not be in the
list above), it raises `ValueError` and never returns `DEAD`.

Details:

- `state` is always a state of the automaton, never `DEAD`. It can be any state, not
  only the start state, and states need not be numbered from 0.
- `token` is an int in `range(vocab.size)`.
- The result of `allowed_tokens` is a list of Python ints and may be empty.
- The automaton is immutable: `advance` returns the new state and changes nothing.
- Learn the automaton only through `step` and `walk`. Do not call its test oracles
  `dfa.accepts`, `dfa.accepts_prefix`, and `dfa.is_live` (the harness rejects them),
  and do not read its private attributes.

## Example

The regular expression `(true|null)` compiles to an automaton with eight states.
State 0 is the start, state 7 is the only accepting state.

![Solid arrows are byte transitions. A dashed arc is one multi-byte token walked from state 0, and it may land in a state that is not accepting.](assets/token-jumps.svg)

With `vocab = ToyVocab(size=64, seed=0)` the relevant tokens are `6 = b't'`,
`19 = b'n'`, `30 = b'e'`, `33 = b'u'`, `40 = b'nul'`, `46 = b'tr'`, `47 = b'null'`,
`49 = b'true'`, `53 = b'nu'`, and `54 = b'tru'`.

```python
dfa = CharDFA.from_regex(r"(true|null)")

allowed_tokens(dfa, 0, vocab)     # [6, 19, 40, 46, 47, 49, 53, 54]
advance(dfa, 0, 46, vocab)        # 4      b'tr': 0 -t-> 2 -r-> 4
allowed_tokens(dfa, 4, vocab)     # [33]   only b'u' can follow "tr"
advance(dfa, 4, 33, vocab)        # 6
allowed_tokens(dfa, 6, vocab)     # [30]   b'e'
advance(dfa, 6, 30, vocab)        # 7      accepting: "true" is complete
allowed_tokens(dfa, 7, vocab)     # []     nothing may follow

allowed_tokens(dfa, 1, vocab)     # [33]   after "n": b'u', but not b'nu' or b'nul'
advance(dfa, 0, 47, vocab)        # 7      b'null' in one token
advance(dfa, 0, 33, vocab)        # raises ValueError: no string starts with "u"
advance(dfa, 7, 1, vocab)         # raises ValueError: token 1 is <|eos|>, no bytes
```

In state 0, eight tokens survive, and only two of them (`b'true'` and `b'null'`) end
in the accepting state. The token `b'fa'` (id 60) dies at its first byte. In state 1
the token `b'nu'` dies at its first byte too: its `n` would have to follow the `n`
already read. In state 7 the text is complete, the list is empty, and the loop's only
legal move is end-of-sequence, which this function does not report.

## Constraints

- `vocab.size` is between 64 and 282 and the automaton has at most a few hundred
  states.
- Results are compared exactly: the same ids in the same (increasing) order, and the
  same states.
- There is no cost limit in this problem. One walk per token is the intended solution;
  the next problem, `dfa-token-index`, is about not repeating that work at every step.

## Why this matters

This function is the core of regex-constrained generation and of the lexer half of
JSON-schema and grammar constraints. Every engine that offers structured output
computes, for the current automaton state, the set of tokens whose bytes keep the text
inside the language. The three kinds of tokens above are exactly where
implementations break: demanding a complete match after every token makes most
outputs impossible to start, testing only the first character lets `": ` through
where only `"` fits, dropping tokens that are not valid UTF-8 on their own makes
`€` unreachable for tokenizers that split it, and letting special tokens through
means a "guaranteed valid" JSON object can end after `{"a`.
