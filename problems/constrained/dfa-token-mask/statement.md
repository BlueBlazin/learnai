A regular expression constrains text one character at a time, whereas a language model generates tokens that can span several characters, part of a character, or no text at all. Given the byte-level deterministic finite automaton (DFA) of a regular expression and its current state, compute which tokens are valid next and advance the automaton when a token is chosen.

![This problem is Allowed tokens and Advance state, found by walking a token's bytes through the automaton.](assets/map.svg)

## Specification

```python
def allowed_tokens(dfa: CharDFA, state: int, vocab: ToyVocab) -> list[int]: ...
def advance(dfa: CharDFA, state: int, token: int, vocab: ToyVocab) -> int: ...
```

**The automaton.** `dfa` is a [`CharDFA`](../../../python/src/learnai/kit/dfa.py), a deterministic automaton that reads bytes. Its states are non-negative integers. Calling `dfa.step(state, byte)` returns the next state after reading a single byte (an integer in `[0, 256)`), or the constant `DEAD` (`-1`) if no string in the language continues with that byte. Calling `dfa.walk(state, data)` reads a `bytes` object sequentially and stops at the first byte that returns `DEAD`. Calling `dfa.is_accepting(state)` reports whether the bytes read so far form a complete string in the language. The automaton is trimmed, meaning every non-`DEAD` state can reach an accepting state by some continuation. Any walk that does not reach `DEAD` therefore represents a valid prefix of a string in the language.

**The vocabulary.** `vocab` is a [`ToyVocab`](../../../python/src/learnai/kit/vocab.py) whose token ids run from `0` to `vocab.size - 1`. Calling `vocab.bytes(token)` returns the `bytes` object for that token. Three kinds of tokens matter here:

- Multi-byte tokens (such as `b'true'` or `b'": '`), which drive the automaton through multiple transitions.
- Partial UTF-8 tokens (such as `b'\xe2\x82'`, the first two of the three bytes of `€`), which end partway through a character.
- Special tokens (such as end-of-sequence), which have the empty byte string `b""`.

**`allowed_tokens(dfa, state, vocab)`** returns the ids, in increasing order, of all tokens `t` such that:

1. `vocab.bytes(t)` has at least one byte, and
2. reading all of its bytes in order from `state` never yields `DEAD`.

The state reached after reading the token does not need to be accepting because later tokens can finish the string. If any byte in the token yields `DEAD`, the entire token is rejected regardless of how many earlier bytes succeeded.

Special tokens fail condition 1 and are never included in the returned list, in any state. Whether generation may end is decided separately by the decoding loop using `dfa.is_accepting(state)`.

**`advance(dfa, state, token, vocab)`** returns the state reached after reading the bytes of `token` from `state`. If `token` is not allowed in `state`, it raises `ValueError` and never returns `DEAD`.

Additional rules:

- `state` is always a valid state of `dfa`, never `DEAD`. It may be any state, not only the start state, and state ids need not start at 0.
- `token` is an integer in `range(vocab.size)`.
- `allowed_tokens` returns a list of Python `int` values, which may be empty.
- `dfa` is immutable. `advance` returns the new state without modifying `dfa`.
- Interact with `dfa` only through `step` and `walk`. Do not call its test oracles (`dfa.accepts`, `dfa.accepts_prefix`, or `dfa.is_live`), which the harness rejects, and do not access its private attributes.

## Example

The regular expression `(true|null)` compiles to an automaton with eight states. State 0 is the start state, and state 7 is the only accepting state.

![Solid arrows are byte transitions. A dashed arc is one multi-byte token walked from state 0, and it may land in a state that is not accepting.](assets/token-jumps.svg)

With `vocab = ToyVocab(size=64, seed=0)`, the relevant tokens are `6 = b't'`, `19 = b'n'`, `30 = b'e'`, `33 = b'u'`, `40 = b'nul'`, `46 = b'tr'`, `47 = b'null'`, `49 = b'true'`, `53 = b'nu'`, and `54 = b'tru'`.

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

In state 0, eight tokens survive, and only two of them (`b'true'` and `b'null'`) land in the accepting state. The token `b'fa'` (id 60) dies on its first byte. In state 1, `b'nu'` also dies on its first byte because another `n` cannot follow the `n` already read. In state 7, the string is complete and no non-empty token can follow, so `allowed_tokens` returns `[]` and the decoding loop can end the sequence.

## Constraints

- `vocab.size` is between 64 and 282, and `dfa` has at most a few hundred states.
- Results are checked for exact equality, including the increasing order of token ids and the exact returned states.
- There is no step-count limit in this problem. Walking each token once per call is the intended approach. The follow-up problem, `dfa-token-index`, removes the repeated work across decoding steps.

## Why this matters

This token-masking step sits at the core of regex-constrained decoding and the lexer stage of JSON-schema and grammar engines. At each decoding step, the engine finds all vocabulary tokens whose bytes keep the generated text inside the target language.

The three token types above are where production implementations commonly break. Requiring an accepting state after every token blocks multi-token outputs from ever starting. Checking only the first byte lets `": ` slip through where only `"` is valid. Rejecting tokens that are not standalone UTF-8 makes characters like `€` unreachable when a tokenizer splits them across tokens. Allowing empty special tokens lets a supposedly valid JSON object terminate mid-key after `{"a`.
