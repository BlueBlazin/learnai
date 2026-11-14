## The idea

Let $L$ be the language of the regular expression and $w$ the bytes generated so far.
The decoder may append a token with bytes $b$ exactly when $w\,b$ is still a prefix of
some string in $L$, because that is the condition under which a valid output remains
reachable. A deterministic automaton summarizes everything about $w$ that matters in
one number, its state $q$, and for a trimmed automaton "is still a prefix" is the same
as "the walk has not died". So the test for a token is a walk of its bytes from $q$:

$$
t \text{ allowed in } q \iff b_t \ne \varepsilon \ \text{ and } \ \delta^*(q, b_t) \ne \text{DEAD},
$$

and the new state is $\delta^*(q, b_t)$. Three consequences follow from this
definition, one for each kind of awkward token.

- **Multi-byte tokens are jumps.** A token is allowed only if *every* byte survives.
  It may cross several transitions, start in the middle of one syntactic element and
  end in the middle of another (`": ` closes a key, adds the colon, and starts the
  space before the value).
- **Acceptance is not the criterion.** After `tr` the text is not in the language, but
  it can become so. Requiring an accepting state after each token would leave only
  the tokens that complete the entire output in one go.
- **Fragments of characters need bytes.** The automaton reads bytes precisely so that
  a token ending inside a UTF-8 character has a state to stop in. Nothing has to
  decode anything.

Special tokens are the one place where the rule needs an explicit clause. Their byte
string is empty, a walk of zero bytes ends where it started, and "did not die" would
let end-of-sequence, padding, and chat markers through in every state. They are
excluded by the "at least one byte" condition, and the right to end is granted
separately, by `is_accepting`.

## Reference solution (Python)

```python
def _walk_token(dfa, state, token, vocab):
    piece = vocab.bytes(token)
    if not piece:
        return DEAD
    return dfa.walk(state, piece)


def allowed_tokens(dfa, state, vocab):
    return [t for t in range(vocab.size) if _walk_token(dfa, state, t, vocab) != DEAD]


def advance(dfa, state, token, vocab):
    after = _walk_token(dfa, state, token, vocab)
    if after == DEAD:
        raise ValueError(f"token {token} is not allowed in state {state}")
    return after
```

`advance` raises instead of returning `DEAD` because a dead state is silent: every
later mask would be empty, and the loop would stop with half an output and no error.

## Common mistakes

- **Requiring acceptance after each token.** `dfa.is_accepting(end)` instead of
  `end != DEAD`. In the worked example this leaves `true` and `null` and removes the
  six tokens that only begin a word.
- **Checking only the first byte** (or first character). For `{"[a-z]+": ...}` the
  token `": ` begins with a legal quote in several states where its second byte is
  illegal. The mask then offers a token that `advance` rejects.
- **Letting empty tokens through.** `dfa.walk(state, b"")` returns `state`. Without
  the explicit exclusion, all five special tokens are allowed everywhere, including
  end-of-sequence in the middle of the structure.
- **Dropping tokens that are not valid UTF-8 alone.** `piece.decode()` fails for
  `b'\xe2\x82'`, and treating that as "unusable" makes every character that the
  tokenizer splits impossible to generate. The same bug appears when the automaton is
  driven with decoded characters instead of bytes.
- **Walking from the start state.** The mask depends on the current state; a mask
  computed from `dfa.start` is right exactly once.
- **Returning `DEAD` from `advance`** instead of raising.

## Complexity

One call walks every token: $O(\sum_t |b_t|)$ automaton steps, which is the total
number of bytes in the vocabulary, at most. For a real tokenizer with 128,000 tokens
that is up to several hundred thousand steps **per generated token**, on the CPU,
while the GPU waits. Since the answer depends only on the state, not on the step, it can be computed
once per state and stored. That is `dfa-token-index`.

## In the wild

- **Outlines** (Willard and Louf, 2023) introduced exactly this formulation: compile
  the regex to a finite-state machine, track its state across tokens, and allow the
  vocabulary strings that the machine can read from the current state. Its current
  Rust core walks each token's bytes through a byte-level DFA, discards the token when
  the walk reaches a dead state, and handles end-of-sequence separately by allowing it
  in final states, as this problem does.
- **llama.cpp** grammars work on code points: for each candidate token it decodes the
  token's bytes, carrying an incomplete UTF-8 sequence over from the previous token,
  and rejects the token if any character cannot be consumed by the grammar. The
  partial-character state is the code-point equivalent of the intermediate byte states
  here.
- **llguidance** and **XGrammar** run the same byte-level test but organize the
  vocabulary as a prefix tree, so that the walk for `tr` is shared by `tru` and
  `true`, and precompute or cache masks per automaton state.
- Hand-built automata can contain *trap* states that are alive but can never reach
  acceptance. For those "did not die" is too weak a test and the real criterion is
  "acceptance is still reachable from the end state". Regex compilers remove such
  states, which is why the simple rule is enough here.
