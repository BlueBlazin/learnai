Write the generation loop of structured output: greedy decoding with a KV cache in
which every step may only choose among the tokens a constraint permits, and
end-of-sequence is permitted only once the output is complete. The constraint tells
you what is allowed; the loop has to ask it at the right moments, keep its state in
step with the tokens, and still run each token through the model only once.

![This problem is the decode loop, which masks, picks, and advances the state once per token.](assets/map.svg)

## Specification

```python
def constrained_greedy(lm, prompt, constraint, max_new_tokens, eos_id): ...
```

`lm` is a [`TinyLM`](../../../python/src/learnai/kit/tinylm.py), used as in
`cached-greedy-decode`: `logits, cache = lm(tokens)` starts a sequence,
`logits, cache = lm(tokens, cache)` continues the cached prefix, and `logits[-1]`
(shape `(vocab_size,)`) scores the next token.

`constraint` is a
[`TokenConstraint`](../../../python/src/learnai/kit/constraint.py). It describes a
set of valid outputs through four methods:

```python
state = constraint.start()                 # the state before any token is generated
constraint.allowed(state)                  # sorted list of token ids that may come next
state = constraint.advance(state, token)   # the state after `token`; ValueError if not allowed
constraint.is_accepting(state)             # True if the output so far is complete
```

States are opaque, hashable values; `advance` returns a new state and changes
nothing. `allowed(state)` may be empty, and it never contains `eos_id`: whether the
output may end is what `is_accepting` says. The constraint covers the generated
tokens only, not the prompt.

**One step.** With the constraint in state $s$, the *permitted* tokens are

$$
\text{permitted}(s) = \text{allowed}(s) \;\cup\; \begin{cases} \{\texttt{eos\_id}\} & \text{if is\_accepting}(s) \\ \emptyset & \text{otherwise.} \end{cases}
$$

The next token is the permitted token with the largest next-token logit (the lowest
id among equal logits). Equivalently: set the logit of every token that is not
permitted to $-\infty$ and take the argmax. Then the constraint is advanced by that
token, unless it is `eos_id`.

**The loop.** Start with $s$ = `constraint.start()` and repeat the step. Generation
ends at the first of:

1. **Nothing is permitted** in the current state: the output cannot be continued and
   may not end. Return what has been generated so far, without EOS. (At the very
   start this returns `[]`.)
2. **`eos_id` is generated.** It is the last element of the result.
3. **`max_new_tokens` tokens** have been generated. The output may then be
   incomplete; that is the caller's problem.

Parameters and result:

- `prompt`: a list of ints, $P \ge 1$ token ids.
- `max_new_tokens`: an int $\ge 1$.
- `eos_id`: an int, a special token of the vocabulary.
- Returns the generated tokens (without the prompt) as a list of Python ints, of
  length $g$ with $0 \le g \le$ `max_new_tokens`.

**Cost limits**, as in `cached-greedy-decode`, for a call that returns $g$ tokens:

| counter | limit | why |
|---|---|---|
| `lm.counters.forward_calls` | $g$ | one call for the prompt, which yields the first token, then one per further token |
| `lm.counters.tokens_processed` | $P + g - 1$, and $0$ if $g = 0$ | each prompt token once, and each generated token once except the last |

Two consequences. The final token is never fed to the model, whichever of the three
rules ended the generation. And the model is not called for a step at which nothing
is permitted: ask the constraint before running the model, so that a dead end costs
no forward call (if nothing is permitted at the start, the model is not called at
all).

## Shapes

| name | shape | dtype |
|---|---|---|
| `prompt` | list of $P$ ints | Python `int` |
| `logits` (from `lm`) | `(T, vocab_size)` for a call with `T` tokens | float32 |
| `constraint.allowed(state)` | sorted list of ints | Python `int` |
| result | list of $g$ ints | Python `int` |

## Example

The constraint is the regular expression `\[\d(, \d)*\]`, a list of digits such as
`[4]` or `[0, 7, 7]`, over the kit's 64-token vocabulary:

```python
lm = TinyLM(fw, seed=5)
vocab = ToyVocab(64, seed=1)
constraint = DFAConstraint(CharDFA.from_regex(r"\[\d(, \d)*\]"), vocab)

constrained_greedy(lm, [0, 17, 42, 5], constraint, 12, eos_id=1)
# [51, 54, 7, 32, 39, 45, 54, 61, 1]      text "[5, 1, 5]", then EOS
# 9 calls, 12 tokens processed
```

| step | text so far | permitted | unconstrained argmax | chosen |
|---|---|---|---|---|
| 1 | (empty) | `[` | 37 `e` | 51 `[` |
| 2 | `[` | the ten digits | 37 `e` | 54 `5` |
| 3 | `[5` | `,` or `, ` or `]` | 37 `e` | 7 `,` |
| 4 | `[5,` | the space | 2 (a special token) | 32 (space) |
| 5 | `[5, ` | the ten digits | 37 `e` | 39 `1` |
| 6 | `[5, 1` | `,` or `, ` or `]` | 33 `.` | 45 `, ` |
| 7 | `[5, 1, ` | the ten digits | 30 (half of an emoji) | 54 `5` |
| 8 | `[5, 1, 5` | `,` or `, ` or `]` | 34 `€` | 61 `]` |
| 9 | `[5, 1, 5]` | EOS | 11 `3` | 1 (EOS) |

Left alone, this model would never produce a list: its favorite token is not
permitted at any step. The mask decides between the few tokens that are. Three
things to notice:

- A token is not a character. After a digit the vocabulary offers both `,` (token 7)
  and the two-character piece `, ` (token 45). Step 3 takes the comma alone, which
  leaves only the space permitted in step 4; step 6 takes both characters at once.
  The loop never looks at text: it trusts `allowed`.
- EOS is not permitted in steps 1 to 8, where the text is not a complete list (at
  step 3, `[5` could be closed, but it is not closed yet). In step 9 the state is
  accepting and no token can extend the text, so EOS is the only permitted token.
  With a pattern like `-?\d+`, EOS and ten digits are permitted together after every
  digit, and the logits decide.
- The step-9 call is still made, although its outcome is forced: 9 tokens, 9 calls,
  and $4 + 8 = 12$ tokens processed. Token 1 is never fed.

## Constraints

- $1 \le P$, $1 \le$ `max_new_tokens`, and $P +$ `max_new_tokens` $\le$
  `lm.max_context` (128).
- The tokens are compared exactly with constrained greedy decoding that uses no
  cache. In the tests the two best permitted logits differ by more than `1e-3`, so
  the result does not depend on how the calls are split, and ties do not occur.
- The counters are compared with the limits above. A solution over a limit fails
  even when its tokens are right.
- Use only the four methods of the constraint. The tests include a constraint that
  is not built from a DFA and whose states are tuples.

## Why this matters

JSON mode, function calling with a schema, regex- and grammar-guided generation all
run this loop. The grammar machinery (a DFA over the vocabulary, a pushdown automaton
for JSON) answers one question, "which tokens may come next?"; the loop turns the
answer into a guarantee by masking the logits before a token is chosen. The
guarantee is exactly as strong as the loop's bookkeeping. If the constraint state
falls behind the tokens, the mask describes the wrong position. If EOS is not
masked, the model can stop inside a string and the caller's JSON parser fails on a
response that was supposed to be valid by construction. If EOS is never offered,
the model keeps emitting digits or whitespace until the token limit. And if validity
is checked after the forward pass instead of before the choice, every rejected
candidate costs a model call, on the hot path of every structured request.
