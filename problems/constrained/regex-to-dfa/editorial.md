## The idea

A regular expression is built from choices: this branch or that one, another
repetition or stop. A deterministic automaton has none. The classical route keeps
the two concerns apart: first build an automaton that is allowed to guess
(a nondeterministic one, where the translation of each construct is a three-line
rule), then remove the guessing mechanically.

**Parsing.** The grammar has four precedence levels, and recursive descent gives
each its own function: `alternation` splits at `|`, `concatenation` collects pieces
until `|` or `)`, `repetition` reads one atom and at most one quantifier, `atom`
reads a literal, an escape, a class, or a parenthesized `alternation`. Precedence
is nothing but which function calls which. Everything an atom can match is reduced
to one node type, "one byte out of this set", so later stages never hear about
classes or escapes.

**Characters over bytes.** The only construct that is not a byte set is "one
character except ...". Written out as a tree, it is an alternation of the ASCII
bytes that are not excluded and the three multi-byte shapes: a lead byte `C2`–`DF`
and one continuation, `E0`–`EF` and two, `F0`–`F4` and three. After that the
automaton knows UTF-8 structurally: after `E2` it is in a state that accepts only
a continuation byte, which is exactly the state a token ending in the middle of `€`
leaves the decoder in.

**Thompson's construction.** Every node becomes a fragment with one entry and one
exit, connected by transitions that read nothing (epsilon moves): a byte set is one
edge; a concatenation chains fragments exit to entry; an alternation fans out from
the entry and back into the exit; `x{m,n}` is `m` copies in a row, then `n - m`
copies, each with a bypass to the exit; `x{m,}` is `m` copies and one copy with a
loop. The fragment count is linear in the size of the pattern with bounds
expanded.

**Subset construction.** After reading some input, the guessing automaton could be
in any of a set of states. That set is the deterministic state. The start is the
epsilon closure of the entry; on a byte, take every edge whose set contains the
byte and close again. Sets are explored breadth-first and numbered as they are
found. A set containing the exit is accepting. A byte that leads nowhere gets no
entry, which is how the result comes out without a dead state: every fragment
state lies on some path to the exit, so every non-empty set can still accept.

**Minimization.** The subset automaton can distinguish histories that the language
does not: for `-?(0|[1-9]\d*)` it has one state for "just read the first digit" and
another for "read more digits", with identical futures. Two states are equivalent
when the same strings are accepted from both. Moore's algorithm computes that
relation from the coarse side: assume all accepting states are equivalent and all
others too, then split any block in which two states disagree on some byte about
the block they move to (or about whether they move at all). Each round either
splits a block or proves the partition stable; a stable partition is the
equivalence. One state per block is the minimal automaton, and by the Myhill–Nerode
theorem it is the same automaton whichever correct construction produced the input,
up to state names. That is why the tests can demand an exact state count.

## Reference solution (Python)

The stages after parsing, in full (the parser is the recursive descent described
above, about a hundred lines, in the reference file):

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

Two details carry weight. In `minimize`, a state's own block is part of its
signature, so a round can only split blocks, never merge them, and "the count did
not grow" means "nothing changed". In `determinize`, the closure is computed once
per distinct target set and not once per byte: for `.` there are 127 ASCII bytes
with the same target.

Other constructions reach the same automaton. Brzozowski derivatives work on the
expression itself (the state after reading a byte is the derivative of the
expression by that byte) and need no nondeterministic stage; with the expressions
normalized well they produce few states directly, and a minimization pass still
guarantees the minimum.

## Common mistakes

- **`{m,n}` off by one.** `n - m + 1` optional copies, or a loop bound that is
  exclusive where the syntax is inclusive. `a{2,4}` must have exactly five states.
- **Wrong precedence.** Splitting at `|` inside the loop that collects pieces makes
  `ab|cd` mean `a(b|c)d`. Applying a quantifier to "everything so far" makes `ab*`
  mean `(ab)*`.
- **Negated classes as byte sets.** `[^"]` as "the 255 other bytes" accepts `C3`
  with nothing after it, and accepts a continuation byte with nothing before it.
  The language then contains byte strings that are not text, and a decoder
  following it can end its output mid-character.
- **`.` matching a newline.** It is "any character except `0A`"; a negated class is
  not restricted that way unless it lists `\n`.
- **Ranges that stop early.** `range(low, high)` drops the last member.
- **`{m,}` as `{m}`**, or `+` as `*`: the loop or the required copy is missing.
- **No minimization.** The language is right and the state count is not. The cost
  appears downstream, in the size of the token index.
- **A dead state.** A complete table with a sink is a correct automaton for
  matching and the wrong object for masking: "the walk is still alive" no longer
  means "this is a prefix of a match", which is the test `dfa-token-mask` relies on.
- **Accepting unsupported syntax.** A compiler that skips the `?` of `a*?` or treats
  `^` as a literal turns a pattern it does not understand into a constraint nobody
  asked for. Refusing is the correct behavior.

## Complexity

Parsing and Thompson's construction are linear in the size of the pattern with
bounds expanded (`x{100}` is a hundred copies). The subset construction is
exponential in the worst case, and the worst case is real: `(a|b)*a(a|b){k}` has
to remember the last $k + 1$ bytes and needs $2^{k+1}$ states, minimal or not. For
the patterns constrained decoding uses (formats with little ambiguity) the number
of sets stays close to the number of fragment states. Each Moore round costs
$O(\text{transitions} \cdot \log)$ for the sorted signatures, and there are at most
as many rounds as states; Hopcroft's algorithm brings the total to
$O(n \log n)$ per byte class and is what production libraries use.

A practical optimization the reference leaves out: bytes that no part of the
pattern distinguishes can be grouped into classes first, so that rows have a dozen
entries, not 256.

## In the wild

The automaton-plus-vocabulary design is from "Efficient Guided Generation for Large
Language Models" (arXiv 2307.09702), the paper behind Outlines: a regular
expression is compiled to a finite-state machine, and the machine's states index
the sets of allowed tokens. The kit's `CharDFA.from_regex`, which the other
problems of this track use, is this problem's function with the same subset and the
same pipeline. Its character table (and this problem's) is deliberately loose: a
strict UTF-8 automaton also excludes overlong encodings (`E0` followed by `80`–`9F`),
surrogates (`ED` followed by `A0`–`BF`), and code points above U+10FFFF (`F4`
followed by `90`–`BF`), at the cost of a few more states.
