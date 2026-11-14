Walking the whole vocabulary through the automaton at every decoding step repeats the
same work again and again: which tokens are allowed depends only on the automaton's
state, not on when it is reached. Build the answers once, as an index from each state
to its allowed tokens and their next states, so that a decoding step is a lookup. The
build itself must not waste work either: tokens that begin with the same bytes share
those steps.

## Specification

```python
def build_index(dfa, vocab): ...
def allowed_from_index(index, state) -> list[int]: ...
def advance_from_index(index, state, token) -> int: ...
```

`dfa` is a [`CharDFA`](../../../python/src/whetstone/kit/dfa.py) and `vocab` a
[`ToyVocab`](../../../python/src/whetstone/kit/vocab.py), as in `dfa-token-mask`, and
"allowed" means the same thing: token `t` is **allowed** in state `q` when
`vocab.bytes(t)` has at least one byte and reading all of its bytes from `q` never
yields `DEAD`. The state reached is its **next state**. Special tokens have no bytes
and are never allowed.

**`build_index(dfa, vocab)`** returns an index that holds, for every state `q` in
`dfa.states`, every token allowed in `q` together with its next state. Conceptually
it is the table

```python
{state: {token: next_state}}       # allowed tokens only; DEAD never appears in it
```

but the representation is yours: the tests only pass the index back to the two query
functions.

**`allowed_from_index(index, state)`** returns the ids of the tokens allowed in
`state`, as a list of Python ints in increasing order (possibly empty).

**`advance_from_index(index, state, token)`** returns the next state of `token` in
`state`, and raises `ValueError` if the token is not allowed there.

`state` is always one of `dfa.states` and `token` is in `range(vocab.size)`. The
queries do not modify the index, and several indexes may exist at once.

### Cost

Every call of `dfa.step(state, byte)` is counted (`dfa.walk` counts one step per byte
it consumes, including the byte that kills the walk). Two limits apply.

**1. Queries are free.** After `build_index` has returned, `allowed_from_index` and
`advance_from_index` make **zero** steps, however often they are called.

**2. The build shares prefixes.** Call a byte string a *prefix* if it is a non-empty
beginning of some token's bytes; the token `b'true'` contributes the prefixes `t`,
`tr`, `tru`, and `true`, and it shares the first three with the tokens `b't'`,
`b'tr'`, and `b'tru'`. A pair (state `q`, prefix `p`) is **live** if `p` without its
last byte can be read from `q` without dying, which is always the case for a prefix of
one byte. `build_index` may make at most

$$
\text{steps} \;\le\; \#\{(q, p) : q \in \texttt{dfa.states},\ (q, p) \text{ is live}\}.
$$

In words: one step per state and *distinct* prefix, and no step along a prefix that
has already died. Walking each token on its own breaks the first half, because the
step for `t` is paid again for `tr`, `tru`, and `true`; continuing to step after
`DEAD` breaks the second. A prefix tree (trie) over the vocabulary, walked once per
state together with the automaton, meets the limit exactly.

As in `dfa-token-mask`, learn the automaton only through `step` and `walk`: the
oracles `dfa.accepts`, `dfa.accepts_prefix`, and `dfa.is_live` are rejected by the
harness, and private attributes are off limits.

## Example

Take the automaton of `(true|null)` from `dfa-token-mask` (eight states, start 0,
accepting 7) and `vocab = ToyVocab(size=64, seed=0)`, where `6 = b't'`, `18 = b'l'`,
`19 = b'n'`, `30 = b'e'`, `33 = b'u'`, `40 = b'nul'`, `43 = b'r'`, `46 = b'tr'`,
`47 = b'null'`, `49 = b'true'`, `53 = b'nu'`, and `54 = b'tru'`. The index holds:

| state | `{token: next_state}` |
|---|---|
| 0 | `{6: 2, 19: 1, 40: 5, 46: 4, 47: 7, 49: 7, 53: 3, 54: 6}` |
| 1 | `{33: 3}` |
| 2 | `{43: 4}` |
| 3 | `{18: 5}` |
| 4 | `{33: 6}` |
| 5 | `{18: 7}` |
| 6 | `{30: 7}` |
| 7 | `{}` |

```python
index = build_index(dfa, vocab)        # dfa.counters.steps == 310
allowed_from_index(index, 0)           # [6, 19, 40, 46, 47, 49, 53, 54]
advance_from_index(index, 0, 46)       # 4
allowed_from_index(index, 4)           # [33]
advance_from_index(index, 1, 53)       # raises ValueError: b'nu' cannot follow "n"
allowed_from_index(index, 7)           # []
                                       # dfa.counters.steps is still 310
```

Where 310 comes from: this vocabulary has 59 tokens with bytes, and their byte strings
have 60 distinct prefixes, 38 of which are single bytes. From state 0 the live pairs
are the 38 single bytes plus `tr`, `tru`, `true`, `nu`, `nul`, and `null`, whose
shorter beginnings survive: 44. In each of the other seven states at most one byte
survives (`u`, `r`, `l`, `u`, `l`, `e`, and none in state 7), and no longer prefix
begins with it, so only the 38 single bytes are live. That gives $44 + 7 \times 38 = 310$. Walking the
59 tokens one by one from each of the eight states takes 484 steps for the same table.

## Constraints

- `vocab.size` is between 64 and 282; the automaton has at most a few hundred states.
- Lists and states are compared exactly. Step counts are exact: there is no tolerance
  on either limit.
- The index may use memory proportional to the number of (state, allowed token)
  pairs.

## Why this matters

The mask is computed on the CPU between two forward passes, once per generated token
and per sequence in the batch. A loop over a 128,000-token vocabulary at every step
can cost more than the model itself for small models, and it stalls the GPU. Moving
the work to compile time, once per regex or schema, is the idea that made
regex-guided generation practical (Willard and Louf, 2023, implemented in Outlines),
and every current engine does some form of it. Compile time is then what users wait
for on the first request with a new schema, which is why the build must share work
across the vocabulary instead of walking each token from scratch.
