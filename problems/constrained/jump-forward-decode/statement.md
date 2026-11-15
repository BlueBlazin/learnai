In structured output most tokens are not decisions. After `{"safe": ` has begun, the
key, the quotes, and the closing brace are dictated by the format, and a forward pass
whose result is masked down to a single token computes nothing. Write the constrained
decoding loop that skips those passes: a token that is the only permitted one is
appended without running the model, and the model is called only when it has a real
choice to make, in one call that also catches it up on everything it skipped. The
output must be exactly what the ordinary constrained loop produces.

## Specification

```python
def jump_forward_decode(lm, prompt, constraint, max_new_tokens, eos_id): ...
```

`lm` is a [`TinyLM`](../../../python/src/whetstone/kit/tinylm.py) and `constraint` a
[`TokenConstraint`](../../../python/src/whetstone/kit/constraint.py), both used as in
`constrained-greedy-decode`:

```python
logits, cache = lm(tokens)  # start a sequence; one row of logits per token
logits, cache = lm(tokens, cache)  # continue the cached prefix with one or more tokens

state = constraint.start()
constraint.allowed(state)  # sorted token ids that may come next; never eos_id
state = constraint.advance(state, token)
constraint.is_accepting(state)  # may the output end here?
```

**The result** is defined by the loop of `constrained-greedy-decode`. With the
constraint in state $s$, the permitted tokens are

$$
\text{permitted}(s) = \text{allowed}(s) \;\cup\; \begin{cases} \{\texttt{eos\_id}\} & \text{if is\_accepting}(s) \\ \emptyset & \text{otherwise,} \end{cases}
$$

the next token is the permitted token with the largest next-token logit given the
prompt and all tokens generated so far, and the constraint is advanced by it unless
it is `eos_id`. Generation ends when nothing is permitted (return what was generated,
without EOS), when `eos_id` has been generated (it is the last element), or when
`max_new_tokens` tokens have been generated, whichever comes first.

**What this problem adds** is how little of the model that takes. Call a step

- **forced** when exactly one token is permitted, and
- **free** when two or more are.

At a forced step the argmax over the permitted tokens can only return that one
token, so its logits are not needed. Note what counts: the *permitted* set. A state
with one allowed token that is also accepting has two permitted tokens (that token
and EOS) and is free. A state that is accepting with nothing allowed has one, EOS,
and is forced: the final EOS of a finished structure costs no model call.

A forced token still belongs to the sequence. The model must have seen it, in its
place, before it scores any later free step, or those logits describe a different
text.

**Cost limits.** Let $g$ be the number of tokens returned, $f$ the number of free
steps among them, and $L$ the number of tokens generated before the last free step
($0 \le L < g$; not defined when $f = 0$).

| counter | limit | why |
|---|---|---|
| `lm.counters.forward_calls` | $f$ | one call per free step; it feeds everything the model has not seen yet (the prompt, or the previous free token and the forced run after it) and its last row scores the choice |
| `lm.counters.tokens_processed` | $P + L$, and $0$ if $f = 0$ | the prompt and every token before the last free step, once each; the last free token and whatever follows it are never fed |

So the prompt does not get a call of its own when the output begins with forced
tokens, a run of forced tokens costs no call however long it is, a forced run at the
end of the output is never fed, and an output that is forced from the first token to
the last costs nothing at all.

Parameters and result:

- `prompt`: a list of ints, $P \ge 1$ token ids. The constraint does not apply to it.
- `max_new_tokens`: an int $\ge 1$. It counts forced tokens like any others: a forced
  run is cut where the limit is reached.
- `eos_id`: an int, a special token of the vocabulary.
- Returns the generated tokens (without the prompt) as a list of Python ints.

> [!NOTE]
> Forced *text* is not a forced *token*. After `{"safe": ` the text `true` or `false`
> must follow, but a vocabulary with the pieces `t`, `tr`, `tru`, and `true` permits
> four tokens that all spell a prefix of `true`. Which of them the model emits is a
> free step like any other: the tokenization is part of the output, and the model's
> later logits depend on it. Only a step with a single permitted token is skipped.

## Shapes

| name | shape | dtype |
|---|---|---|
| `prompt` | list of $P$ ints | Python `int` |
| `logits` (from `lm`) | `(T, vocab_size)` for a call with `T` tokens | float32 |
| `constraint.allowed(state)` | sorted list of ints | Python `int` |
| result | list of $g$ ints | Python `int` |

## Example

The constraint is the regular expression `\{"safe": (true|false)\}` over the kit's
64-token vocabulary:

```python
lm = TinyLM(fw, seed=5)
vocab = ToyVocab(64, seed=1)
constraint = DFAConstraint(CharDFA.from_regex(r'\{"safe": (true|false)\}'), vocab)

jump_forward_decode(lm, [0, 17, 42, 5], constraint, 24, eos_id=1)
# [52, 12, 46, 25, 29, 37, 43, 50, 46, 37, 40, 1]      text '{"safe": false}', then EOS
# 3 calls, 11 tokens processed
```

| step | text so far | permitted | kind | chosen | model call |
|---|---|---|---|---|---|
| 1 | (empty) | `{` or `{"` | free | 52 `{` | the prompt, 4 tokens |
| 2 | `{` | `"` | forced | 12 `"` | |
| 3 | `{"` | `s` | forced | 46 `s` | |
| 4 | `{"s` | `a` | forced | 25 `a` | |
| 5 | `{"sa` | `f` | forced | 29 `f` | |
| 6 | `{"saf` | `e` | forced | 37 `e` | |
| 7 | `{"safe` | `"` or `":` or `": ` | free | 43 `": ` | `[52, 12, 46, 25, 29, 37]`, 6 tokens |
| 8 | `{"safe": ` | `t` `tr` `tru` `true` `f` `fa` `fal` `fals` `false` | free | 50 `fal` | `[43]`, 1 token |
| 9 | `{"safe": fal` | `s` | forced | 46 `s` | |
| 10 | `{"safe": fals` | `e` | forced | 37 `e` | |
| 11 | `{"safe": false` | `}` | forced | 40 `}` | |
| 12 | `{"safe": false}` | EOS | forced | 1 | |

Twelve tokens, three decisions, three calls ($f = 3$), and $4 + 7 = 11$ tokens
processed ($L = 7$: seven tokens precede the choice of `fal`). The ordinary loop
returns the same twelve tokens with 12 calls and 15 tokens processed.

- Step 1 is free although the text must begin with `{"`: the vocabulary has both
  `{` and `{"`, and the model prefers `{`. After that choice the quote is forced.
- Step 7's call feeds the first free token and the five forced tokens after it
  together. Its six rows of logits score six positions; only the last one is used.
- Steps 9 to 12 are never fed. No decision comes after them.

## Constraints

- $1 \le P$, $1 \le$ `max_new_tokens`, and $P +$ `max_new_tokens` $\le$
  `lm.max_context` (128).
- The tokens are compared exactly with constrained greedy decoding that uses no
  cache and makes one call per step. In the tests the two best permitted logits of
  every free step differ by more than `1e-3`, so the result does not depend on how
  the calls are split, and ties do not occur.
- The counters are compared with the limits above. A solution over a limit fails
  even when its tokens are right.
- Use only the four methods of the constraint. The tests include a constraint that
  is not built from a DFA and whose states are tuples.

## Why this matters

A JSON schema with fixed keys is mostly punctuation and key names. An engine that
runs one forward pass per token spends most of a structured request computing
distributions it then masks to one entry; an engine that knows a token is forced
appends it for free and batches the run into the next pass, where extra positions in
one call are far cheaper than extra calls. SGLang calls this jump-forward decoding
and llguidance fast-forward tokens. The trap is the token boundary. Jumping over
forced *text* means choosing its tokenization for the model, and a model shown `{`
then `"` where it would have produced `{"` continues from a sequence it has rarely
seen. And a loop that appends forced tokens to its output but not to the model's
context returns valid JSON whose free choices were made about a different document.
