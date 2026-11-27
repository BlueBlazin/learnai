## The idea

`prompt-lookup-draft` finds where the last few tokens appeared earlier by scanning the context for a fixed-length n-gram. At scale, that breaks down in two ways. Scanning costs $O(L)$ per decoding step, and capping the n-gram length discards longer matches that carry much stronger signal. Removing the length cap only makes scanning slower. A suffix automaton solves both issues because it maintains the longest earlier suffix match as part of its core structure.

**States.** For any substring $w$ of the sequence, let $\operatorname{endpos}(w)$ be the set of indices where an occurrence of $w$ ends. Two substrings with the same end set are equivalent, and each state of the suffix automaton represents one equivalence class (plus a root state for the empty string). All strings in a single state are suffixes of one another with consecutive lengths, so a state is characterized by its longest string of length `length[v]`. Following `next` transitions from the root along the tokens of a substring leads to its state.

**Suffix links.** Take the longest string in state $v$ and trim tokens from its left end. Initially the shorter suffixes still have the same end set and belong to $v$. Eventually a shorter suffix also occurs elsewhere in the sequence, so its end set strictly grows. The state of that first shorter suffix is `link[v]`. In other words, `link[v]` points to the state of the *longest suffix of $v$'s strings that ends at more positions than $v$ does*.

**The match is a suffix link.** Let `last` be the state representing the entire sequence of length $L$. Its end set is just $\{L - 1\}$. Therefore `link[last]` is the state of the longest suffix that ends at $L - 1$ and at least one other position. Because no index is greater than $L - 1$, that other position must be earlier, which is the exact definition of the longest earlier match (overlaps included). Hence

$$
m = \texttt{length}[\texttt{link}[\texttt{last}]] .
$$

No search happens at query time because the automaton already maintains this link.

**The earliest occurrence.** Store `first_end[v]`, the smallest index in state $v$'s end set. Because new tokens are only appended at the right end of the sequence, a state's earliest end position never changes once the state is created:

- A new state created for the token at index `pos` has end set $\{\text{pos}\}$, so `first_end[cur] = pos`.
- A cloned state splits off shorter strings that an existing state $q$ already held, so those strings first ended wherever $q$'s strings first ended: `first_end[clone] = first_end[q]`.

Since every string in a state shares the same end set, the earliest occurrence of the match of length $m$ ends at `first_end[link[last]]`. The draft therefore starts at `first_end[link[last]] + 1`.

**Online construction.** Appending token $t$ at index `pos` creates a new state `cur` with `length[cur] = pos + 1` and `first_end[cur] = pos`. Walk suffix links starting from the previous `last` to visit all suffixes of the old sequence from longest to shortest. Each state lacking a transition on $t$ gets a transition to `cur`. If the walk falls off the root, `link[cur]` becomes the root.

Otherwise the walk stops at the first state $p$ that already has a transition on $t$ to some state $q$. The longest string of $p$ followed by $t$ is the longest suffix of the new sequence that appeared earlier. If `length[q] == length[p] + 1`, then every string in $q$ is a suffix of the new sequence, so `link[cur] = q`. Otherwise $q$ also contains longer strings that are *not* suffixes of the new sequence, and its end set must split. Create a `clone` state with `length[clone] = length[p] + 1`, copying $q$'s `next` transitions, `link`, and `first_end`. Redirect transitions on $t$ from $p$ and its suffix-link ancestors that pointed to $q$ so they point to `clone`, and set both `link[q]` and `link[cur]` to `clone`.

## Reference solution (Python)

```python
class SuffixIndex:
    def __init__(self):
        self._tokens = []
        self._next, self._link, self._length, self._first_end = [{}], [-1], [0], [-1]
        self._last = 0

    def _state(self, length, first_end, link, nxt):
        self._next.append(nxt)
        self._link.append(link)
        self._length.append(length)
        self._first_end.append(first_end)
        return len(self._length) - 1

    def extend(self, token):
        nxt, link, length = self._next, self._link, self._length
        pos = len(self._tokens)
        self._tokens.append(token)
        cur = self._state(pos + 1, pos, 0, {})
        p = self._last
        while p != -1 and token not in nxt[p]:
            nxt[p][token] = cur
            p = link[p]
        if p != -1:
            q = nxt[p][token]
            if length[q] == length[p] + 1:
                link[cur] = q
            else:
                clone = self._state(length[p] + 1, self._first_end[q], link[q], dict(nxt[q]))
                while p != -1 and nxt[p].get(token) == q:
                    nxt[p][token] = clone
                    p = link[p]
                link[q] = clone
                link[cur] = clone
        self._last = cur

    def match_length(self):
        return self._length[self._link[self._last]] if self._tokens else 0

    def draft(self, max_len, min_match=1):
        if not self._tokens:
            return []
        state = self._link[self._last]
        if self._length[state] < min_match:
            return []
        start = self._first_end[state] + 1
        return self._tokens[start : start + max_len]
```

Because the root state has length `0`, the "no earlier match" case needs no special handling beyond checking whether `_tokens` is empty. When no non-empty suffix matched, `link[last]` is the root and `0 < min_match`.

## Common mistakes

- **Searching the history for every draft.** Even an optimized history scan (binary searching on the monotone match length with a fast C-level substring search) takes $O(L \log L)$ per draft and runs about 50x slower than the automaton at 12,000 tokens, worsening as context grows.
- **Recording the current position on a clone.** Setting `first_end[clone] = pos` is tempting because the clone is created while appending index `pos`. However, the clone's strings already occurred earlier and first ended at `first_end[q]`. Using `pos` copies from a later occurrence, or even past the end of the sequence.
- **Leaving a clone's transitions empty.** The cloned state represents strings that were already followed by tokens earlier in the sequence. Starting it with `{}` instead of `dict(nxt[q])` loses those continuations and shortens future matches.
- **Never splitting states.** Linking `cur` directly to `q` when `length[q] > length[p] + 1` overstates the match length because `length[q]` counts longer strings that are not suffixes of the current sequence.
- **Drafting from a shorter match.** Walking suffix links down to the shortest state with length at least `min_match` throws away longer context. `min_match` is a minimum threshold on the longest match, not a target length.
- **Returning the most recent occurrence.** The specification requires the earliest occurrence. Storing the earliest end position also takes $O(1)$ per state because it never changes, whereas the latest occurrence changes on every append.
- **Off-by-one errors.** The draft starts *after* the earlier occurrence (`first_end + 1`), and a match of length equal to `min_match` qualifies (`< min_match` returns `[]`).

## Complexity

For a sequence of length $L \ge 3$, a suffix automaton has at most $2L - 1$ states and $3L - 4$ transitions. `extend` runs in amortized $O(1)$ dictionary operations (by the standard potential argument on `length[link[last]]`), plus copying one transition dictionary per clone, which is bounded by the number of distinct tokens following that context. `match_length` is $O(1)$ and `draft` is $O(\text{max\_len})$. Memory is $O(L)$ states. By contrast, scanning the history costs $O(L)$ per draft for a fixed n-gram and $O(L \cdot m)$ for an unbounded match.

## In the wild

- The online algorithm is the classical linear-time suffix automaton (directed acyclic word graph) construction of Blumer et al. (1985), which SAM-Decoding also uses for its static corpus automaton.
- "SAM Decoding: Speculative Decoding via Suffix Automaton" (arXiv:2411.10666) uses this exact structure as a retrieval drafter. Each state stores its suffix link, transitions, length, and `min_endpos` (the earliest end position), and clones inherit `min_endpos` from the original state. SAM-Decoding maintains a static automaton over a reference corpus alongside a dynamic automaton over the prompt and generated tokens, using match length to pick between them and a neural drafter. Its dynamic automaton tracks a matching state across accepted tokens ("transfer, then expand"), which corresponds to `link[last]` right after insertion here.
- Prompt lookup decoding is built into Hugging Face `transformers` via `prompt_lookup_num_tokens` in `generate` (with `max_matching_ngram_size`, defaulting to 2), and vLLM includes n-gram and suffix-based proposers among its speculative decoding backends. Because the target model always verifies proposed drafts (as in `greedy-speculative-verify`), index bugs do not change the generated tokens, only the decoding speed.
