## The idea

The constrained loop chooses each token as an argmax over the permitted set, and the
permitted set is a function of the constraint state alone. When it has one element
the choice is made before the model runs: every logit but one is masked to
$-\infty$, so the forward pass decides nothing. Jump-forward decoding takes that
token directly.

Skipping the *call* is not skipping the *token*. The model's distribution at a later
free step is conditioned on the whole sequence, forced tokens included, so they must
enter the KV cache before that step. But nothing needs their logits, which removes
the reason to feed them one at a time: they can wait and go in together with the
token before them, in the single call that the next free step needs anyway. A
causal model computes position $i$ from positions $\le i$ only, so one call with
$k$ tokens leaves the cache in the same state as $k$ calls with one token, and its
last row is the distribution the ordinary loop would have seen.

That gives the invariant of the loop: a list `pending` holds exactly the tokens the
model has not seen, in order. It starts as the prompt, grows by every generated
token, and is emptied by a call. A call happens only at a free step. So:

- calls $= f$, the number of free steps;
- tokens processed $= P + L$, everything before the last free step, because what is
  pending when the loop ends is dropped;
- the result is token for token the result of the ordinary loop, since at forced
  steps both take the only permitted token and at free steps both take an argmax
  over the same logits.

"Forced" has to be judged on the permitted set, EOS included. One allowed token in an
accepting state is a decision (continue or stop). No allowed token in an accepting
state is not: EOS is forced, and the closing call of the ordinary loop disappears.

## Reference solution (PyTorch)

```python
def jump_forward_decode(lm, prompt, constraint, max_new_tokens, eos_id):
    state = constraint.start()
    generated = []
    cache = None
    pending = list(prompt)
    while len(generated) < max_new_tokens:
        permitted = list(constraint.allowed(state))
        if constraint.is_accepting(state):
            permitted.append(eos_id)
        if not permitted:
            break
        if len(permitted) == 1:
            token = permitted[0]
        else:
            logits, cache = lm(pending, cache)
            pending = []
            keep = [False] * lm.vocab_size
            for candidate in permitted:
                keep[candidate] = True
            masked = torch.where(torch.tensor(keep), logits[-1], float("-inf"))
            token = int(torch.argmax(masked))
        generated.append(token)
        if token == eos_id:
            break
        state = constraint.advance(state, token)
        pending.append(token)
    return generated
```

The JAX and MLX versions differ only in the names of `where` and `argmax`.

## Common mistakes

- **Jumping over forced text instead of forced tokens.** After `{"safe": fal` the
  text `se}` is certain, and so is `{"` at the start. But where the vocabulary
  offers both `{` and `{"`, two tokens are permitted and the model's preference
  between them is part of the result. Taking the longest piece (or re-encoding the
  forced string with the tokenizer) is a different algorithm with a different
  output. It can be the right engineering choice, see below, but it is not what the
  masked argmax computes.
- **Forced tokens appended to the output but not to the model's input.** The output
  is still valid, which is why this survives casual testing. Every free choice after
  the first forced run is made from a context that lacks it.
- **One call per forced token.** Taking the token without reading the logits and
  then feeding it immediately "to keep the cache current" saves an argmax and no
  model work.
- **A call for the prompt alone.** If the first step is forced, the prompt's logits
  are never read. Let the prompt wait in `pending` like any other unseen token.
- **Feeding the tail.** After the last free step nothing is chosen, so the last
  free token and the forced run behind it need not be fed. Flushing `pending` after
  the loop costs a call for nothing (unless the caller wants the cache, which this
  function does not return).
- **Counting `allowed` instead of `permitted`.** `len(allowed) == 1` jumps over a
  state where the model could also have stopped; requiring "not accepting" for a
  jump pays a call for a forced EOS.
- **An inner jump loop that forgets the token limit.** Forced tokens count toward
  `max_new_tokens`; a `while forced:` loop needs the same bound as the outer one.

## Complexity

$f$ forward calls and $P + L$ tokens processed, against $g$ and $P + g - 1$ for the
ordinary loop, plus one `allowed`, one `is_accepting`, and at most one `advance` per
token. The saving is in calls, not in positions: a forced token before the last free
step still costs one position of attention. On a GPU a call with a few extra
positions costs about as much as a call with one, so the wall-clock gain is close to
the fraction of forced steps, which for JSON with a fixed schema is most of them.

## In the wild

- **SGLang** introduced the name. Its description (the paper "SGLang: Efficient
  Execution of Structured Language Model Programs", arXiv 2312.07104, and the LMSYS
  blog post "Fast JSON Decoding for Local LLMs with Compressed Finite State Machine")
  works at the level of text: consecutive singular transitions of the regex automaton
  are compressed into one edge, and the engine prefills that edge's string instead of
  decoding it. Because the string, not a token sequence, is what is forced, SGLang
  re-tokenizes the whole text after appending it, so that the model continues from
  the tokenizer's own segmentation; the post reports that different tokenizations of
  the jumped text change the later distributions, and puts the cost of
  re-tokenizing at about 4 percent.
- **llguidance** (the constraint engine behind Guidance) reports forced tokens to
  the host as fast-forward tokens: after a token is committed, the parser returns
  any further tokens the grammar forces, and the host appends them all. It enables
  this only for tokenizers with a canonical tokenization.
- The two differ exactly at the note in the statement. Token-level forcing, as in
  this problem, reproduces the masked argmax exactly and jumps less often. Text-level
  forcing jumps over every forced string and has to decide how to tokenize it. A
  related repair at the start of generation is `token-healing`.
- With sampling instead of argmax nothing changes: a distribution masked to one
  token is a point mass, so a forced step consumes no randomness worth modeling.
