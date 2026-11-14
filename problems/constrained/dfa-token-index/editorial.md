## The idea

The mask for a state is a pure function of the state. A decoding run visits the same
states over and over (inside a JSON string the automaton sits in one state for most
tokens), and different requests with the same schema visit the same states too. So
compute the function for every state once and store it:

$$
\text{index}[q] = \{\, t \mapsto \delta^*(q, b_t) \;:\; b_t \ne \varepsilon,\ \delta^*(q, b_t) \ne \text{DEAD} \,\}.
$$

Storing the next state next to each token makes `advance` a lookup as well, so after
the build the automaton is never consulted again. The automaton over bytes has become
an automaton over tokens: same states, one transition per allowed token.

That settles the queries. The build is where the work now is: $|Q|$ states times the
whole vocabulary. Walking each token separately repeats steps, because vocabularies
are full of tokens that extend other tokens: `t`, `tr`, `tru`, `true`. From a given
state the walk of `true` passes through exactly the states that the walks of `t`,
`tr`, and `tru` end in. A trie over the tokens' bytes makes that sharing explicit:
each node is a distinct prefix, and walking the trie depth-first while carrying the
automaton state visits each prefix once per state. When a step dies, every token
below that node is dead too, and the whole subtree is skipped without a single
further step. For restrictive states, which are the common case in structured
formats, almost the entire vocabulary is cut off at the first byte.

## Reference solution (Python)

```python
class _Node:
    def __init__(self):
        self.children = {}     # next byte -> node
        self.tokens = []       # tokens whose bytes end exactly here


def _vocab_trie(vocab):
    root = _Node()
    for token in range(vocab.size):
        piece = vocab.bytes(token)
        if not piece:
            continue                       # special tokens are never allowed
        node = root
        for byte in piece:
            node = node.children.setdefault(byte, _Node())
        node.tokens.append(token)
    return root


def build_index(dfa, vocab):
    root = _vocab_trie(vocab)
    index = {}
    for state in dfa.states:
        moves = {}
        stack = [(root, state)]
        while stack:
            node, at = stack.pop()
            for byte, child in node.children.items():
                after = dfa.step(at, byte)
                if after == DEAD:
                    continue               # prune the subtree
                for token in child.tokens:
                    moves[token] = after
                stack.append((child, after))
        index[state] = moves
    return index


def allowed_from_index(index, state):
    return sorted(index[state])


def advance_from_index(index, state, token):
    try:
        return index[state][token]
    except KeyError:
        raise ValueError(f"token {token} is not allowed in state {state}") from None
```

The trie is built once and reused for every state. A node keeps a *list* of tokens
because real vocabularies occasionally contain two ids with identical bytes.

Two other ways to stay within the budget are worth knowing. Sorting the tokens by
their bytes puts tokens with a common prefix next to each other, and a walk that
keeps the states of the previous token's bytes can resume after the shared part: the
same sharing without an explicit tree. And since the count is of `dfa.step` calls,
remembering every `(state, byte)` result in a table of your own also meets it. That
is fair, it is what a real DFA already is, but note what it leaves untouched: the
loop over every byte of every token for every state is still there, only cheaper per
iteration. The trie removes iterations.

## Common mistakes

- **A lazy "index".** Keeping the automaton and walking at query time, or filling the
  entry for a state on its first query, is a cache, not an index: the first visit to
  each state still stalls a decoding step, and the stall lands in the middle of a
  request. The zero-step check catches both.
- **Walking each token separately.** Correct output, 484 steps instead of 310 in the
  example, and a worse ratio on real vocabularies, where long tokens are common.
- **An index keyed by token only.** Recording where each token leads from the start
  state answers the first step and no other. The key is the pair (state, token).
- **Storing dead entries.** Writing `moves[token] = after` before testing `after`
  puts `token: DEAD` into the table. Everything that survives its first bytes and dies
  on its last then counts as allowed, and `advance` hands `DEAD` to the caller.
- **Tokens only at the leaves.** In a trie of tokens, interior nodes are tokens too:
  `t` and `tr` sit on the path to `true`. Recording a token only where a branch ends
  loses exactly the short, frequent ones.
- **No pruning.** Stepping on from `DEAD` (the kit lets you: the result stays `DEAD`)
  walks the entire trie for every state. The index is right and the budget is blown.
- **Special tokens at the root.** Inserting a token with empty bytes leaves it at the
  root, "reached" from every state without a step. End-of-sequence is the decoding
  loop's decision, made with `dfa.is_accepting(state)`.

## Complexity

Let $N$ be the number of trie nodes (distinct prefixes; at most the total number of
bytes in the vocabulary) and $|Q|$ the number of states. The build makes at most
$|Q| \cdot N$ steps, and in practice far fewer because of pruning. The index holds one
entry per (state, allowed token) pair, $O(|Q| \cdot |V|)$ in the worst case; a state
such as "inside a string", where nearly every token is allowed, costs a full row. A
query is a dict lookup, plus a sort of the allowed ids unless the sorted list is
stored at build time. Engines store each row as a bitmask of $|V|$ bits, ready to be
applied to the logits.

## In the wild

- **Willard and Louf (2023)**, "Efficient Guided Generation for Large Language
  Models" (arXiv:2307.09702), is the source of the idea. Section 3 frames regex
  guidance as transitions of a finite-state machine and builds a map $\sigma$ from
  each FSM state to the set of vocabulary strings the machine can read from that
  state (their Algorithms 3 and 4: for each vocabulary string, find the states from
  which it can be walked). With the map in a hash table, a step's mask is a lookup
  whose cost does not depend on the vocabulary size, where the earlier approach paid
  $O(|V|)$ per token. The paper's map stores token sets; storing the next state with
  each token, as here, is what the implementation does.
- **Outlines** is that implementation. Its Rust core (`outlines-core`) builds
  `state -> {token id -> next state}` by taking each reachable state in turn and
  walking every token's bytes through a byte-level DFA, dropping walks that reach a
  dead state, and it adds an end-of-sequence transition to every final state. It does
  not use a vocabulary trie: its build is the "each token separately" loop of this
  problem's naive solution.
- **llguidance** keeps the vocabulary as a trie laid out flat in memory and walks it
  against the lexer when a mask is needed, pruning dead subtrees as above; its
  documentation puts a trie node visit at about 13 CPU cycles and a simple regex mask
  for a 128k-token vocabulary at about 1.5 ms. It computes masks on the fly instead
  of precomputing every state, which keeps startup cheap.
- **XGrammar** (Dong et al., 2024) precomputes, for each grammar position, the tokens
  whose validity does not depend on the parser stack, and checks only the remaining
  context-dependent tokens at run time.
- A context-free grammar has no finite set of states to index, which is why the
  grammar engines above mix precomputation with run-time checks. The
  `json-pushdown-validator` problem shows where the finite-state picture ends.
