Implement `jump_forward_decode`, a constrained greedy decoding loop that skips model forward passes whenever only one token is permitted. Forced tokens are appended directly without calling the model, then fed together in a single batch the next time the model has a real choice. The returned token sequence must match standard constrained greedy decoding exactly.

![This problem is the decode loop, which here skips the model when only one token is allowed.](assets/map.svg)

## Specification

```python
def jump_forward_decode(lm, prompt, constraint, max_new_tokens, eos_id): ...
```

`lm` is a [`TinyLM`](../../../python/src/learnai/kit/tinylm.py) and `constraint` is a [`TokenConstraint`](../../../python/src/learnai/kit/constraint.py), used as in `constrained-greedy-decode`:

```python
logits, cache = lm(tokens)         # start a sequence; one row of logits per token
logits, cache = lm(tokens, cache)  # continue the cached prefix with one or more tokens

state = constraint.start()
constraint.allowed(state)          # sorted token ids that may come next (never eos_id)
state = constraint.advance(state, token)
constraint.is_accepting(state)     # whether the output may end here
```

The other parameters are:

- `prompt`: initial sequence of $P \ge 1$ token ids. The constraint does not apply to the prompt.
- `max_new_tokens`: maximum number of tokens to generate ($\ge 1$). Forced tokens count toward this limit just like any other token.
- `eos_id`: special token id marking the end of generation.

The function returns the generated tokens (excluding `prompt`) as a list of Python ints.

### Decoding semantics

The output matches the loop in `constrained-greedy-decode`. When the constraint is in state $s$, the set of permitted next tokens is

$$
\text{permitted}(s) = \text{allowed}(s) \;\cup\; \begin{cases} \{\texttt{eos\_id}\} & \text{if is\_accepting}(s) \\ \emptyset & \text{otherwise.} \end{cases}
$$

At each step, the next token is the permitted token with the highest next-token logit given the prompt and all tokens generated so far. Unless that token is `eos_id`, the constraint advances to `constraint.advance(state, token)`. Generation stops as soon as one of three conditions holds:

- `permitted(s)` is empty (return the tokens generated so far, without `eos_id`),
- `eos_id` is generated (included as the final element), or
- `max_new_tokens` tokens have been generated.

### Forced steps and cost limits

Classify each decoding step by the size of `permitted(s)`:

- A **forced** step has exactly one permitted token.
- A **free** step has two or more permitted tokens.

At a forced step, the argmax over `permitted(s)` can only return that single token, so the model's logits are not needed. Notice that this depends on the *permitted* set, not just `allowed(s)`. A state with one allowed token that is also accepting has two permitted tokens (that token and `eos_id`), making it a free step. Conversely, an accepting state with no allowed tokens has only `eos_id` permitted, so the final `eos_id` is forced and costs no model call.

Even though a forced token skips a forward pass, it still belongs to the sequence. The model must process every forced token in its proper position before scoring any later free step, or the logits will correspond to the wrong text.

Let $g$ be the number of returned tokens, $f$ be the number of free steps among them, and $L$ be the number of generated tokens that precede the last free step ($0 \le L < g$, undefined when $f = 0$). Your implementation must stay within both counter limits:

| counter | limit | explanation |
|---|---|---|
| `lm.counters.forward_calls` | $f$ | One call per free step, feeding all unseen tokens (either the prompt, or the previous free token plus any forced run after it) so the last row of logits scores the choice. |
| `lm.counters.tokens_processed` | $P + L$ (or $0$ if $f = 0$) | The prompt and every token before the last free step are each fed once. The last free token and any trailing forced tokens are never fed. |

In particular, the prompt does not trigger its own call when generation starts with forced tokens, a run of forced tokens costs no calls regardless of length, trailing forced tokens are never fed, and a completely forced output makes zero model calls.

> [!NOTE]
> Forced *text* is not the same as a forced *token*. After `{"safe": `, the text `true` or `false` must follow, yet a vocabulary containing `t`, `tr`, `tru`, and `true` permits multiple tokens that spell prefixes of `true`. Choosing among them is a free step because the tokenization becomes part of the output and affects later logits. Only skip a step when a single token is permitted.

## Shapes

| name | shape | dtype |
|---|---|---|
| `prompt` | list of $P$ ints | Python `int` |
| `logits` (from `lm`) | `(T, vocab_size)` for a call with `T` tokens | float32 |
| `constraint.allowed(state)` | sorted list of ints | Python `int` |
| result | list of $g$ ints | Python `int` |

## Example

Using the regular expression `\{"safe": (true|false)\}` over the kit's 64-token vocabulary:

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

This produces 12 tokens with 3 free steps ($f = 3$) and $4 + 7 = 11$ tokens processed ($L = 7$, since seven tokens precede the choice of `fal`). Standard constrained decoding returns the same 12 tokens using 12 calls and 15 tokens processed.

- Step 1 is free even though the text must begin with `{"`. The vocabulary contains both `{` and `{"`, and the model prefers `{`, after which `"` becomes forced.
- Step 7 feeds the first free token together with the five forced tokens that followed it. The call returns six rows of logits, and only the last row is used.
- Steps 9 through 12 are never fed because no decision follows them.

## Constraints

- $1 \le P$, $1 \le$ `max_new_tokens`, and $P +$ `max_new_tokens` $\le$ `lm.max_context` (128).
- Output tokens are compared exactly against uncached single-step constrained greedy decoding. In all tests, the top two permitted logits at every free step differ by more than `1e-3`, so ties do not occur and batching calls does not change the result.
- Both `lm.counters.forward_calls` and `lm.counters.tokens_processed` must stay within the limits above. Exceeding either limit fails even if the output tokens match.
- Use only the four methods on `constraint`. The test suite includes a constraint that is not backed by a DFA and uses tuples as states.

## Why this matters

A JSON schema with fixed keys consists mostly of punctuation and field names. An engine that runs one forward pass per token spends most of its time computing probability distributions only to mask them down to a single entry. By appending forced tokens for free and batching them into the next forward pass, serving engines eliminate most decode calls because processing a few extra positions in one pass is far cheaper than launching separate passes.

SGLang calls this technique jump-forward decoding, and llguidance calls it fast-forward tokens. The subtle failure modes lie at token boundaries and cache state. Jumping over forced *text* instead of forced *tokens* imposes a tokenization on the model, which can push it onto a rare token sequence like `{` followed by `"` instead of `{"`. Even worse, appending forced tokens to the output without feeding them into the model's KV cache still produces syntactically valid JSON, while every subsequent free choice is conditioned on a different document.
