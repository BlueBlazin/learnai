## The idea

Let $L$ be the language of the regular expression and $w$ be the bytes generated so far. The decoder may append a token $t$ with byte string $b_t$ if and only if $w b_t$ is a prefix of at least one string in $L$, because that guarantees a valid completion still exists.

A deterministic automaton summarizes the history $w$ in a single state $q$. Let $\varepsilon$ denote the empty byte string, and let $\delta^*(q, b)$ denote the state reached by feeding the bytes of $b$ into the automaton starting from $q$ (or $\text{DEAD}$ if any transition fails). Because the automaton is trimmed, every non-$\text{DEAD}$ state can still reach an accepting state. Testing whether $w b_t$ remains a valid prefix therefore reduces to walking $b_t$ from $q$:

$$
t \text{ allowed in } q \iff b_t \ne \varepsilon \ \text{ and } \ \delta^*(q, b_t) \ne \text{DEAD}.
$$

When $t$ is allowed, the next state is $\delta^*(q, b_t)$. Three properties follow directly from this rule:

- **Multi-byte tokens span multiple transitions.** Every byte in $b_t$ must survive. A single token can close one syntactic element and begin another, as `": ` closes a JSON key, adds the colon, and starts the space before the value.
- **Intermediate states do not need to be accepting.** After reading `tr`, the text is not yet in $L$, but later tokens can complete `true`. Requiring an accepting state after each token would allow only tokens that complete the entire string at once.
- **Partial UTF-8 tokens work automatically at the byte level.** Because the automaton operates on bytes rather than decoded characters, a token ending partway through `€` simply lands in an intermediate byte state without requiring UTF-8 decoding.

Special tokens need the explicit $b_t \ne \varepsilon$ clause. Walking zero bytes from $q$ returns $q$ without dying, which would allow end-of-sequence, padding, and chat markers in every state. Excluding empty tokens keeps the mask strictly about text continuation, while `dfa.is_accepting(state)` separately controls whether generation may stop.

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

`advance` raises `ValueError` instead of returning `DEAD` so invalid moves fail loudly. Returning `DEAD` would silently cause every subsequent mask to be empty, halting decoding mid-output with no error.

## Common mistakes

- **Requiring acceptance after each token.** Checking `dfa.is_accepting(end)` instead of `end != DEAD` keeps only `true` and `null` in the worked example and drops the six tokens that begin a word.
- **Checking only the first byte or character.** For `{"[a-z]+": ...}`, the token `": ` starts with a valid quote in states where its second byte is illegal. The mask then permits a token that `advance` rejects.
- **Allowing empty tokens.** Calling `dfa.walk(state, b"")` returns `state`. Without checking `if not piece`, all special tokens pass in every state, allowing end-of-sequence in the middle of an incomplete output.
- **Rejecting tokens that are not standalone UTF-8.** Calling `piece.decode()` fails on fragments like `b'\xe2\x82'`. Dropping those tokens, or running the automaton on decoded characters instead of bytes, makes split characters impossible to generate.
- **Walking from `dfa.start` instead of `state`.** The allowed set depends on the current state, so walking from the start state only works on the first step.
- **Returning `DEAD` from `advance`.** Disallowed tokens must raise `ValueError`.

## Complexity

Let $|b_t|$ be the byte length of token $t$. Each call to `allowed_tokens` walks every token in the vocabulary, taking $O(\sum_t |b_t|)$ automaton steps in the worst case (at most the total number of bytes across all vocabulary tokens). For a 128,000-token vocabulary, that means hundreds of thousands of CPU steps per generated token while the GPU waits. Because the mask depends only on `state`, it can be precomputed once per state and stored in an index, which is the goal of `dfa-token-index`.

## In the wild

- **Outlines** (Willard and Louf, 2023) popularized this formulation by compiling a regex to a finite-state machine, tracking its state across generated tokens, and allowing tokens whose bytes can be read from the current state. Its Rust core walks each token's bytes through a byte-level DFA, discards tokens that reach a dead state, and allows end-of-sequence separately in accepting states.
- **llama.cpp** runs grammars on Unicode code points. For each candidate token, it decodes the token's bytes while carrying any incomplete UTF-8 sequence from the previous token, rejecting the token if any resulting character cannot be consumed. That carried partial-character state plays the same role as intermediate byte states here.
- **llguidance** and **XGrammar** perform the same byte-level check over a prefix tree of the vocabulary, sharing the walk for `tr` across `tru` and `true`, and cache or precompute masks per automaton state.
- Hand-built automata can contain trap states that are not `DEAD` yet can never reach an accepting state. In an untrimmed automaton, checking `end != DEAD` is not enough, and you must check whether an accepting state is still reachable. Regex compilers trim unreachable and trap states automatically, so checking against `DEAD` suffices here.
