## The idea

A regular expression describes choices, such as picking between branches or deciding whether to repeat an atom. A deterministic finite automaton (DFA) cannot make guesses. The classical compilation pipeline separates these two concerns by first building a nondeterministic finite automaton (NFA) that is allowed to guess via empty (epsilon) transitions, then determinizing and minimizing it mechanically.

**Parsing.** The grammar has four precedence levels, so recursive descent maps each level to its own function. `alternation` splits on `|`, `concatenation` collects pieces until `|` or `)`, `repetition` reads an atom followed by an optional quantifier, and `atom` reads a literal, escape, character class, or parenthesized `alternation`. Precedence falls directly out of this call hierarchy. Every basic atom reduces to a single node type representing "one byte from this set", so downstream stages do not need to handle classes or escapes.

**Characters over bytes.** The only constructs that do not reduce to a single byte set are the "any character except ..." forms (`.`, `[^...]`, `\D`, `\W`, `\S`). Each expands into an alternation of the allowed ASCII bytes and the three multi-byte UTF-8 shapes: a lead byte in `C2`–`DF` plus one continuation byte (`80`–`BF`), `E0`–`EF` plus two continuation bytes, and `F0`–`F4` plus three continuation bytes. This encodes UTF-8 directly into the automaton structure. After reading `E2`, the DFA enters a state that only accepts a continuation byte, which matches what happens when an LLM token ends in the middle of `€`.

**Thompson's construction.** Each AST node becomes an NFA fragment with one entry state and one exit state, linked by epsilon transitions that consume no input. A byte set is a single labeled edge. Concatenation chains fragments from exit to entry. Alternation fans out from the new entry to each branch and merges their exits into the new exit. A bounded repetition `x{m,n}` chains `m` required copies followed by `n - m` optional copies, each with an epsilon bypass to the exit. An unbounded repetition `x{m,}` chains `m` required copies followed by one looping copy. The number of NFA states is linear in the size of the pattern after expanding bounds.

**Subset construction.** After reading a byte prefix, an NFA could be in any subset of its states. Each reachable subset becomes a single DFA state. The DFA start state is the epsilon closure of the NFA entry. From any subset and any byte, follow all NFA edges matching that byte and take the epsilon closure of the targets. Exploring subsets breadth-first ensures every DFA state is reachable. A DFA state is accepting if its subset contains the NFA exit. Omitting transitions for bytes that lead to no NFA states automatically avoids creating a dead state, because every NFA fragment state lies on a path to the exit and therefore every non-empty subset can still reach an accepting state.

**Minimization.** The subset DFA can keep distinct states that have identical future behavior. In `-?(0|[1-9]\d*)`, it creates separate states for "just read the first digit" and "read more digits", even though both are accepting and loop on all digits. Two states are equivalent if they accept the exact same remaining byte strings. Moore's partition refinement algorithm starts with two blocks (accepting and non-accepting) and repeatedly splits any block whose states disagree on which block a byte transitions to, or whether a transition exists at all. When a full pass splits no blocks, the partition is stable and defines the equivalence classes. Merging each block into one state yields the minimal DFA. By the Myhill–Nerode theorem, this minimal DFA is unique up to state numbering regardless of how the input DFA was built, which is why the tests can check for an exact state count.

## Reference solution (Python)

Here are the NFA, determinization, and minimization stages in full. The recursive-descent parser described above is about a hundred lines in the reference file.

```python
class NFA:
    def __init__(self):
        self.edges = []  # edges[s]: list of (byte set, target)
        self.free = []  # free[s]: targets reachable without reading a byte

    def state(self):
        self.edges.append([])
        self.free.append([])
        return len(self.edges) - 1

    def build(self, node):
        entry, exit_ = self.state(), self.state()
        kind = node[0]
        if kind == "set":
            self.edges[entry].append((node[1], exit_))
        elif kind == "cat":
            here = entry
            for part in node[1]:
                first, last = self.build(part)
                self.free[here].append(first)
                here = last
            self.free[here].append(exit_)
        elif kind == "alt":
            for part in node[1]:
                first, last = self.build(part)
                self.free[entry].append(first)
                self.free[last].append(exit_)
        else:
            _, part, low, high = node
            here = entry
            for _ in range(low):
                first, last = self.build(part)
                self.free[here].append(first)
                here = last
            if high is None:
                first, last = self.build(part)
                self.free[here] += [first, exit_]
                self.free[last] += [first, exit_]
            else:
                for _ in range(high - low):
                    first, last = self.build(part)
                    self.free[here] += [first, exit_]
                    here = last
                self.free[here].append(exit_)
        return entry, exit_

    def closure(self, states):
        seen, stack = set(states), list(states)
        while stack:
            for target in self.free[stack.pop()]:
                if target not in seen:
                    seen.add(target)
                    stack.append(target)
        return frozenset(seen)


def determinize(nfa, entry, exit_):
    start = nfa.closure([entry])
    number = {start: 0}
    table = {}
    queue = deque([start])
    while queue:
        subset = queue.popleft()
        moves = {}
        for state in subset:
            for byte_set, target in nfa.edges[state]:
                for byte in byte_set:
                    moves.setdefault(byte, set()).add(target)
        row = table[number[subset]] = {}
        seen = {}  # many bytes lead to the same target set: close each set once
        for byte, targets in moves.items():
            key = frozenset(targets)
            if key not in seen:
                reached = nfa.closure(key)
                if reached not in number:
                    number[reached] = len(number)
                    queue.append(reached)
                seen[key] = number[reached]
            row[byte] = seen[key]
    accepting = {n for subset, n in number.items() if exit_ in subset}
    return table, accepting


def minimize(table, accepting):
    block = {state: int(state in accepting) for state in table}
    count = len(set(block.values()))
    while True:
        signature = {
            state: (block[state], tuple(sorted((byte, block[t]) for byte, t in row.items())))
            for state, row in table.items()
        }
        ids = {}
        block = {state: ids.setdefault(sig, len(ids)) for state, sig in signature.items()}
        if len(ids) == count:
            break
        count = len(ids)
    merged = {block[s]: {byte: block[t] for byte, t in row.items()} for s, row in table.items()}
    return merged, block[0], {block[s] for s in accepting}
```

Two implementation details are worth noting. In `minimize`, including a state's current block ID in its signature guarantees that each round can only split blocks and never merge them, so stopping when `len(ids) == count` proves convergence. In `determinize`, caching closures by `frozenset(targets)` avoids recomputing the same closure for every byte in a large set, such as the 127 ASCII bytes matched by `.`.

Other constructions produce the same minimal DFA. Brzozowski derivatives differentiate the regex directly with respect to each byte and skip the NFA stage. With good expression normalization they stay compact directly, though a minimization pass is still needed to guarantee the exact minimum.

## Common mistakes

- **`{m,n}` off by one.** Creating `n - m + 1` optional copies or treating `n` as exclusive breaks bounded repetitions. `a{2,4}` must have exactly five states.
- **Wrong precedence.** Splitting on `|` inside the concatenation loop turns `ab|cd` into `a(b|c)d`. Applying a quantifier to the accumulated concatenation turns `ab*` into `(ab)*`.
- **Negated classes as raw byte sets.** Implementing `[^"]` as the other 255 bytes accepts a lone `C3` lead byte or an isolated continuation byte. The automaton then accepts invalid UTF-8 and allows a decoder to stop mid-character.
- **`.` matching a newline.** `.` must exclude the newline byte `0A`. Negated classes do match `\n` unless `\n` is listed inside them.
- **Exclusive character ranges.** Using `range(low, high)` omits the upper endpoint `high`.
- **Broken unbounded quantifiers.** Compiling `{m,}` as `{m}` or `+` as `*` drops either the loop or the required copy.
- **Skipping minimization.** The automaton accepts the right language, but redundant states fail the exact state-count check and bloat downstream token indexes.
- **Keeping a dead sink state.** A complete transition table with a sink state matches strings correctly, but breaks token masking because `dfa-token-mask` assumes any existing transition leads to a live prefix of a valid match.
- **Silently accepting unsupported syntax.** Ignoring the `?` in `a*?` or treating `^` as a literal silently compiles a different constraint than requested. Raise `ValueError` instead.

## Complexity

Parsing and Thompson's construction run in linear time in the size of the pattern with repetition bounds expanded (for example, `x{100}` creates 100 copies). Subset construction is exponential in the worst case. For example, `(a|b)*a(a|b){k}` must track the last $k + 1$ bytes and requires $2^{k+1}$ states even after minimization. For typical structured-output patterns with low ambiguity, the number of subset states stays close to the NFA size. With $n$ DFA states and $m$ transitions, each round of Moore's algorithm takes $O(m \log m)$ time to sort signatures, for at most $n$ rounds. Production libraries use Hopcroft's algorithm, which runs in $O(n \log n)$ time per byte class.

A practical optimization omitted in the reference is grouping bytes into equivalence classes before determinization. Bytes that no part of the pattern distinguishes share transitions, shrinking each row from up to 256 entries to a few dozen.

## In the wild

This automaton-plus-vocabulary pipeline comes from "Efficient Guided Generation for Large Language Models" (arXiv 2307.09702), the paper behind Outlines. A regular expression compiles into a DFA, and each DFA state indexes the set of allowed tokens. The kit's `CharDFA.from_regex` used by other problems in this track implements this exact subset and pipeline. Its UTF-8 byte table (and this problem's) is intentionally permissive. A strict UTF-8 automaton also rejects overlong encodings (`E0` followed by `80`–`9F`), surrogates (`ED` followed by `A0`–`BF`), and code points above U+10FFFF (`F4` followed by `90`–`BF`), at the cost of a few extra states.
