## The idea

`prompt-lookup-draft` answers "where did the last few tokens occur before" by
scanning the context, for a bounded n-gram. Two things are wrong with that at
scale: the scan costs time proportional to the context on every step, and the bound
throws away the best evidence there is, a long match. Dropping the bound makes the
scan worse, not better. The suffix automaton removes both problems, because the
answer to the question is something it maintains anyway.

**States.** For a substring $w$ of the sequence, let $\operatorname{endpos}(w)$ be
the set of positions at which an occurrence of $w$ ends. Two substrings with the
same end set are equivalent; the states of the suffix automaton are the
equivalence classes (plus the empty string as the root). The strings of one state
are suffixes of each other with consecutive lengths, so a state is described by
its longest string, of length `length[v]`. Reading a substring from the root along
the `next` transitions arrives at its state.

**Suffix links.** Take the longest string of state $v$ and shorten it from the
left. For a while the end set stays the same (those are the other strings of $v$);
at some length it grows, because the shorter string also occurs elsewhere. The
state of that first shorter string is `link[v]`. So `link[v]` is the state of the
*longest suffix of $v$'s strings that ends at more positions than they do*.

**The match is a suffix link.** Let `last` be the state of the whole sequence. Its
end set is the single position $L - 1$. Its link is therefore the state of the
longest suffix that ends at $L - 1$ **and somewhere else**, and somewhere else can
only be earlier: that is the definition of the match, overlaps included. Hence

$$
m = \texttt{length}[\texttt{link}[\texttt{last}]] .
$$

No search happens at query time; the construction has already done it.

**The earliest occurrence.** Store `first_end[v]`, the smallest element of
$v$'s end set. It never changes after the state is created, because positions are
only ever added at the end of the sequence:

- a state created for the token at position `pos` has end set $\{pos\}$;
- a clone takes over strings that the original state $q$ already held, and they
  ended first wherever $q$'s strings ended first: `first_end[clone] = first_end[q]`.

Since all strings of a state share the end set, the earliest occurrence of the
match of length $m$ ends at `first_end[link[last]]`, and the draft starts at the
next index.

**Construction.** Appending token $t$ creates one state `cur` for the new longest
string. Walking the suffix links from the old `last` visits the suffixes of the old
sequence from longest to shortest; each one that was never followed by $t$ gets a
transition to `cur`. The walk stops at the first state $p$ that already has a
transition on $t$, to some state $q$. The string (longest of $p$) + $t$ is the
longest suffix of the new sequence that occurred before. If it is the longest
string of $q$ (`length[q] == length[p] + 1`), $q$ is the link of `cur`. Otherwise
$q$ also holds longer strings that are *not* suffixes of the new sequence, and its
end set is about to split: the strings up to length `length[p] + 1` now also end at
the new position, the longer ones do not. The clone takes the shorter strings (and
$q$'s transitions and link), the transitions from $p$ and its link ancestors that
led to $q$ are redirected to it, and both $q$ and `cur` link to it.

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

The root has length 0, so "no match" needs no special case beyond the empty index:
`link[last]` is the root, and $0 <$ `min_match`.

## Common mistakes

- **Searching the history for every draft.** Correct, and the reason this problem
  exists. Even the best version of it (the match length is monotone, so a binary
  search over the suffix length with a C-speed substring search per probe) costs
  $O(L \log L)$ per draft and measures about 50x slower than the automaton at
  12,000 tokens, growing with the context.
- **A clone that records the current position.** `first_end[clone] = pos` looks
  natural, since the clone is created while appending position `pos`. But the
  clone's strings are old; they ended first where the original's did. The draft
  is then copied from a later occurrence, or, when the recorded end is the last
  position, from past the end of the sequence.
- **A clone without the original's transitions.** The clone holds strings that
  were already followed by tokens; starting it with an empty `next` forgets those
  continuations, and later matches come out too short.
- **Never splitting.** Linking `cur` to $q$ unconditionally overstates the match:
  `length[q]` counts strings that are not suffixes of the sequence.
- **Drafting from a shorter match.** Following links down to the shortest state
  that still satisfies `min_match` anchors the copy on less context than is
  available. `min_match` is a threshold on the longest match, not a target.
- **The most recent occurrence.** The specification asks for the earliest one,
  which is also the one a state can remember in constant space; the latest
  occurrence changes with every append.
- **Off by one.** The copy starts *after* the earlier occurrence
  (`first_end + 1`), and a match of exactly `min_match` tokens qualifies.

## Complexity

The automaton has at most $2L - 1$ states and $3L - 4$ transitions for a sequence
of length $L \ge 3$. `extend` is amortized $O(1)$ dictionary operations (the
classical argument: the length of `link[last]` grows by at most one per token and
every step of either loop shortens it or a related potential), plus the copy of
one transition table per clone, which is bounded by the number of distinct tokens
that follow a context. `match_length` is $O(1)$ and `draft` is $O(\text{max\_len})$.
Memory is $O(L)$ states. For comparison, the n-gram scan is $O(L)$ per draft for a
fixed n-gram and $O(L \cdot m)$ for an unbounded one.

## In the wild

- The online construction is the classical linear-time one for the suffix
  automaton, also known as the directed acyclic word graph (Blumer et al., 1985,
  the algorithm the SAM-Decoding paper builds its static automaton with).
- "SAM Decoding: Speculative Decoding via Suffix Automaton" (arXiv:2411.10666)
  uses exactly this structure as a retrieval drafter. Each state stores its link,
  its transitions, its length, and `min_endpos`, the earliest end position; the
  draft is the tokens after `min_endpos` of the matched state, and a clone copies
  the original's `min_endpos`. The paper keeps a *static* automaton over a text
  corpus and a *dynamic* one over the prompt and the generated tokens, and uses
  the match length to choose between them and a model-based drafter. Its dynamic
  automaton keeps a matching state that it moves with each accepted token before
  inserting that token ("transfer, then expand"), following suffix links on a
  mismatch. That state is the same one this problem reads off as
  `link[last]` after the insertion: the longest suffix of the new sequence that is
  a substring of the old one.
- Prompt lookup decoding itself is in `transformers` as the
  `prompt_lookup_num_tokens` argument of `generate` (with
  `max_matching_ngram_size`, 2 when unset), and vLLM ships n-gram and suffix-based
  proposers among its speculative decoding methods. Whatever proposes the draft,
  the target model verifies it as in `greedy-speculative-verify`, so the output
  never depends on the index being right; only the speed does.
