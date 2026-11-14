## The idea

A language model assigns a score to every token at every step. A constraint
partitions the vocabulary, at every step, into tokens that keep the output inside
the language and tokens that do not. Constrained decoding removes the second group
*before* a token is chosen, so the choice is the best token among the valid ones.
Since each chosen token keeps the output a valid prefix, the output is valid at
every length, by induction; and since EOS is offered only in accepting states, an
output that ends with EOS is a complete member of the language.

The loop carries two pieces of state that must describe the same sequence: the KV
cache (what the model has seen) and the constraint state (where the output is in the
language). Each generated token updates both, the constraint immediately through
`advance`, the model lazily when the next step needs logits.

End-of-sequence is the one token the constraint does not list, because it has no
text. The loop adds it to the permitted set exactly when `is_accepting` holds. That
single rule yields three behaviors: EOS is impossible in the middle of the
structure; EOS competes with the other tokens where the output may end or go on
(`12` is a complete number, and so is `123`); and EOS is forced where the output is
complete and nothing can follow.

The remaining case, nothing allowed and not accepting, cannot occur for a
well-formed constraint over a vocabulary that covers the language. It does occur in
practice when the vocabulary lacks a needed character or a grammar has a hole, and a
loop that calls argmax on an all-masked row then emits token 0 forever. Stopping is
the honest outcome.

## Reference solution (PyTorch)

```python
def constrained_greedy(lm, prompt, constraint, max_new_tokens, eos_id):
    state = constraint.start()
    generated = []
    cache = None
    pending = prompt
    while len(generated) < max_new_tokens:
        permitted = list(constraint.allowed(state))
        if constraint.is_accepting(state):
            permitted.append(eos_id)
        if not permitted:
            break
        logits, cache = lm(pending, cache)
        keep = [False] * lm.vocab_size
        for token in permitted:
            keep[token] = True
        masked = torch.where(torch.tensor(keep), logits[-1], float("-inf"))
        token = int(torch.argmax(masked))
        generated.append(token)
        if token == eos_id:
            break
        state = constraint.advance(state, token)
        pending = [token]
    return generated
```

The cost limits follow from the order of the loop body. The permitted set depends
only on the constraint state, so it is known before the model runs; checking it
first means a dead end costs nothing. Feeding a token only at the top of the next
iteration means the last token, whatever ended the generation, is never fed. So $g$
tokens cost $g$ calls and $P + g - 1$ tokens, as without a constraint: the mask is
free in model work. The JAX and MLX versions differ
only in the names of `where` and `argmax`. The boolean mask always has the shape of
the vocabulary, whatever the number of permitted tokens, which also keeps JAX from
compiling a new gather for every list length.

## Common mistakes

- **Not advancing the state.** The first token is right, and every later step is
  masked as if it were the first: `[[[[[`. Advancing a *copy* and discarding it, or
  advancing only on some branches, fails the same way.
- **EOS always permitted.** Leaving EOS out of the mask (or adding it
  unconditionally) lets the model stop wherever it likes. Models like to stop: the
  result is truncated JSON that was promised to be valid.
- **EOS never permitted, or only when nothing else is.** Masking with `allowed`
  alone gives a loop that cannot end a number or a string of unbounded length and
  runs to the token limit; permitting EOS only when forced does the same for every
  language in which a complete output can be extended.
- **Multiplying by the mask.** `logits * mask` is the probability-space operation
  applied to logits: a forbidden token gets logit 0, which beats every permitted
  token with a negative logit. Masking logits means `-inf`.
- **Validating after the fact.** Choosing the unconstrained argmax, running the
  model on it, and only then asking the constraint turns every rejection into a
  wasted forward call (173 calls instead of 9 in the worked example). Checking the
  candidate *before* feeding it and retrying with the next-best logit is correct and
  costs no model work, which is the difference between the two orders.
- **Running the model before the dead-end check.** One wasted call per dead end, and
  with an empty permitted set at the start, a whole prefill for an empty answer.
- **Advancing on EOS.** `advance(state, eos_id)` raises `ValueError`: EOS is not a
  token of the language. Check for EOS first.

## Complexity

$g$ forward calls and $P + g - 1$ tokens processed, plus one `allowed` and at most
one `advance` per step. Building the mask is $O(V)$ per step here. With a real
vocabulary of 100,000 or more tokens, computing `allowed` is the expensive part,
which is why engines precompute it per automaton state (`dfa-token-index`) and store
masks as bitsets.

## In the wild

- Outlines (Willard and Louf, 2023) compiles a regex or JSON schema to a finite
  automaton and indexes, for every state, the tokens that keep it alive; each step is
  then a dictionary lookup and a mask, as with `DFAConstraint` here.
- vLLM's structured outputs choose among several backends (xgrammar, guidance,
  outlines, lm-format-enforcer); each produces a token bitmask per step that the
  engine applies to the logits before sampling.
- llama.cpp's sampler takes the other order by default as an optimization: it
  samples a token from the unconstrained distribution, checks that one token against
  the grammar, and only if it is rejected applies the grammar to all candidates and
  samples again. No model call is repeated, since the logits are already there.
- Greedy choice under a mask is local. It yields the best *next* valid token, not
  the most probable valid string, and a mask can push the model into a continuation
  it finds unlikely; engines accept that for the guarantee. The token limit remains
  outside the guarantee: a response cut by `max_tokens` can be invalid JSON under
  any constraint.
- Steps with a single permitted token need no logits at all. Skipping the forward
  call for such forced tokens and feeding them together is jump-forward decoding
  (`jump-forward-decode`).
