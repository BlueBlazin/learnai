Walking the entire vocabulary through a deterministic finite automaton (DFA) at every decoding step repeats the same work, because the set of allowed tokens depends only on the current state. Precompute an index that maps each state to its allowed tokens and their next states so decoding queries become lookups. During the build, share work across tokens that begin with the same bytes.

![This problem is Token index, built once so that a decoding step is a lookup.](assets/map.svg)

## Specification

```python
def build_index(dfa: CharDFA, vocab: ToyVocab): ...
def allowed_from_index(index, state: int) -> list[int]: ...
def advance_from_index(index, state: int, token: int) -> int: ...
```

`dfa` is a [`CharDFA`](../../../python/src/learnai/kit/dfa.py) and `vocab` is a [`ToyVocab`](../../../python/src/learnai/kit/vocab.py), as in `dfa-token-mask`. A token `t` is **allowed** in state `q` when `vocab.bytes(t)` has at least one byte and reading all of its bytes in order from `q` never yields `DEAD`. The state reached after reading those bytes is its **next state**. Special tokens have empty byte strings and are never allowed.

**`build_index(dfa, vocab)`** builds and returns an index containing, for every state `q` in `dfa.states`, every token allowed in `q` along with its next state. Conceptually, it represents the nested table:

```python
{state: {token: next_state}}       # allowed tokens only; DEAD never appears in it
```

You may choose any internal representation, because the tests only pass the returned index to `allowed_from_index` and `advance_from_index`.

**`allowed_from_index(index, state)`** returns the ids of all tokens allowed in `state` as a list of Python `int` values in increasing order (possibly empty).

**`advance_from_index(index, state, token)`** returns the next state for `token` in `state`, or raises `ValueError` if `token` is not allowed in `state`.

In all calls, `state` is in `dfa.states` and `token` is in `range(vocab.size)`. Queries must not mutate `index`, and multiple indexes may coexist at the same time.

### Cost

Every call to `dfa.step(state, byte)` is counted, and `dfa.walk` counts one step per byte consumed (including the byte that yields `DEAD`). Two strict limits apply:

1. **Queries make zero steps.** After `build_index` returns, `allowed_from_index` and `advance_from_index` must make 0 calls to `dfa.step`, no matter how many times they are called.
2. **The build shares prefixes and prunes dead paths.** Call a non-empty byte string a *prefix* if some token's bytes start with it. For example, `b'true'` contributes the prefixes `t`, `tr`, `tru`, and `true`, sharing the first three with `b't'`, `b'tr'`, and `b'tru'`. A pair `(q, p)` of a state `q` and a prefix `p` is **live** if `p` without its last byte can be read from `q` without reaching `DEAD` (which is always true for 1-byte prefixes). `build_index` may make at most:

$$
\text{steps} \;\le\; \#\{(q, p) : q \in \texttt{dfa.states},\ (q, p) \text{ is live}\}.
$$

This bound allows one `dfa.step` per state and distinct prefix, with no steps taken below a prefix that has already reached `DEAD`. Walking each token independently exceeds the bound because it repeats the step for `t` across `t`, `tr`, `tru`, and `true`. Continuing to step after hitting `DEAD` also exceeds it. Building a prefix tree (trie) over the vocabulary and walking it once per state alongside `dfa` meets the bound exactly.

As in `dfa-token-mask`, interact with `dfa` only through `step` and `walk`. Do not call `dfa.accepts`, `dfa.accepts_prefix`, or `dfa.is_live`, and do not access private attributes.

## Example

Take the `(true|null)` automaton from `dfa-token-mask` (eight states, start state 0, accepting state 7) and `vocab = ToyVocab(size=64, seed=0)`. The relevant tokens are `6 = b't'`, `18 = b'l'`, `19 = b'n'`, `30 = b'e'`, `33 = b'u'`, `40 = b'nul'`, `43 = b'r'`, `46 = b'tr'`, `47 = b'null'`, `49 = b'true'`, `53 = b'nu'`, and `54 = b'tru'`. The index stores:

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

![Nine of these tokens as a trie, walked from state 0. Each arrow is one `dfa.step`, and a byte that reaches `DEAD` gets no entry.](assets/vocab-trie.svg)

This vocabulary has 59 non-empty tokens containing 60 distinct prefixes, 38 of which are single bytes. From state 0, the live pairs are the 38 single bytes plus `tr`, `tru`, `true`, `nu`, `nul`, and `null` (whose shorter beginnings survive), giving 44 live pairs. In each of the other seven states, at most one single byte survives (`u`, `r`, `l`, `u`, `l`, `e`, and none in state 7), and no longer prefix starts with any of them, leaving only the 38 single-byte prefixes live. In total, $44 + 7 \times 38 = 310$ steps. Walking all 59 tokens separately from each of the eight states takes 484 steps for the exact same table.

## Constraints

- `vocab.size` is between 64 and 282, and `dfa` has at most a few hundred states.
- Returned lists and states are compared for exact equality. Step counts are checked strictly with zero tolerance on both limits.
- The index may use memory proportional to the number of `(state, allowed token)` pairs.

## Why this matters

Constrained decoding computes a token mask on the CPU between forward passes, once per generated token for every sequence in the batch. Looping over a 128,000-token vocabulary at every step can take longer than the forward pass of a small model and leave the GPU idle.

Moving this work to compile time—once per regex or JSON schema—is what made regex-guided generation fast enough for production (Willard and Louf, 2023, in Outlines). Because users still wait on compile time for the first request with a new schema, sharing prefix work across the vocabulary keeps that startup cost low.
