## The idea

For a given automaton and vocabulary, the allowed token mask depends only on the current state. A single generation visits the same states repeatedly (for instance, staying in one state for most tokens inside a JSON string), and separate requests using the same schema revisit the same states as well.

Let $Q$ be the set of automaton states, $b_t$ be the byte string of token $t$, $\varepsilon$ be the empty byte string, and $\delta^*(q, b)$ be the state reached by reading the bytes of $b$ in order from state $q$ (or $\text{DEAD}$ if any byte fails). Precomputing the allowed tokens and their destination states for every state $q \in Q$ gives the table:

$$
\text{index}[q] = \{\, t \mapsto \delta^*(q, b_t) \;:\; b_t \ne \varepsilon,\ \delta^*(q, b_t) \ne \text{DEAD} \,\}.
$$

Storing each token's next state makes both `allowed_from_index` and `advance_from_index` pure lookups, so the byte-level automaton is never called again after `build_index`. In effect, the byte-level DFA has been compiled into a token-level DFA over the same states, with one transition per allowed token.

The main cost is now building that table across all $|Q|$ states. Walking each token from scratch repeats work whenever tokens share a prefix, such as `t`, `tr`, `tru`, and `true`. From any state $q$, the walk for `true` visits the exact same states where `t`, `tr`, and `tru` end. Organizing the vocabulary into a prefix tree (trie) makes that sharing explicit. Each trie node represents a distinct prefix, so a depth-first walk of the trie paired with the current DFA state steps each prefix at most once per state. Whenever a transition reaches `DEAD`, every token in that subtree is also dead, so the entire subtree is pruned immediately. In restrictive states (the common case in structured outputs), almost the entire vocabulary is pruned at the first byte.

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

The trie is constructed once and reused across all states. Each node stores a list of tokens rather than a single token id because real vocabularies sometimes assign two token ids to the same byte string.

Two alternative approaches also satisfy the step budget. First, sorting tokens lexicographically by their bytes places shared prefixes next to each other, so a walk that keeps the previous token's state stack can resume after the longest common prefix without building an explicit tree. Second, memoizing `(state, byte)` transitions in a local cache avoids duplicate `dfa.step` calls, though it still loops over every byte of every token for every state instead of pruning dead subtrees like a trie does.

## Common mistakes

- **Building a lazy cache instead of an index.** Keeping `dfa` and walking tokens on demand, or populating `index[state]` on the first query for `state`, stalls decoding mid-request and fails the zero-step query requirement.
- **Walking each token independently.** This produces the right table but blows the step budget (484 steps instead of 310 in the example, and far worse on real vocabularies with longer tokens).
- **Keying the index only by token.** Recording transitions only from the start state works for the first decoding step and fails for every other state. The index must be keyed by `(state, token)`.
- **Storing `DEAD` transitions.** Recording `moves[token] = after` before checking `if after == DEAD` inserts dead tokens into the table, causing them to appear in `allowed_from_index` and causing `advance_from_index` to return `DEAD` instead of raising `ValueError`.
- **Recording tokens only at trie leaves.** Interior trie nodes can also be complete tokens, such as `t` and `tr` on the path to `true`. Recording tokens only at leaves drops short, frequent tokens.
- **Forgetting to prune dead subtrees.** Continuing to step after `after == DEAD` (which `dfa.step` allows, returning `DEAD` again) walks the entire trie for every state and blows the step budget.
- **Inserting empty special tokens at the root.** Putting a `b""` token on the root node makes it appear reachable from every state with zero steps. End-of-sequence is handled separately by the decoding loop via `dfa.is_accepting(state)`.

## Complexity

Let $|Q|$ be the number of automaton states, $|V|$ be the vocabulary size, and $N$ be the number of trie nodes (the number of distinct token prefixes, at most the total number of bytes in the vocabulary). Building the index takes at most $|Q| \cdot N$ steps, and far fewer in practice thanks to dead-subtree pruning.

The index stores one entry per `(state, allowed token)` pair, which is $O(|Q| \cdot |V|)$ in the worst case. Permissive states, such as "inside a JSON string," can fill nearly an entire row of size $|V|$. Each `advance_from_index` call is an $O(1)$ dictionary lookup, and `allowed_from_index` is a lookup plus a sort unless the sorted id list is cached during `build_index`. Production engines typically store each row as a $|V|$-bit mask that can be applied directly to model logits.

## In the wild

- **Willard and Louf (2023)**, "Efficient Guided Generation for Large Language Models" (arXiv:2307.09702), introduced this indexing strategy. Section 3 compiles a regex into a finite-state machine and precomputes a map $\sigma$ from each state to the vocabulary strings readable from that state, turning per-step masking from $O(|V|)$ into a constant-time lookup. The paper stores token sets, while implementations also store the next state alongside each token.
- **Outlines** (`outlines-core` in Rust) implements this as `state -> {token id -> next state}` by walking every token's bytes from each reachable state through a byte-level DFA, discarding dead walks, and adding an end-of-sequence transition on accepting states. It does not use a vocabulary trie, so its build resembles the per-token loop.
- **llguidance** stores the vocabulary as a flat trie in memory and walks it against the lexer on demand, pruning dead subtrees just as above. Its documentation reports roughly 13 CPU cycles per trie node visit and about 1.5 ms for a simple regex mask over a 128k-token vocabulary, keeping startup cheap by avoiding full precomputation.
- **XGrammar** (Dong et al., 2024) precomputes, for each grammar position, the tokens whose validity does not depend on the parser stack, leaving only context-dependent tokens to be checked at runtime.
- Context-free grammars do not have a finite set of states to index completely, which is why grammar engines combine precomputation with runtime checks. The `json-pushdown-validator` problem explores where finite-state indexing stops.
