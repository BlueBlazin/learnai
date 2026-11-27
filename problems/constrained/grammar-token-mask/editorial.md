## The idea

**From one stack to a set of stacks.** A grammar with no alternatives can be matched with a single stack of symbols remaining to be read. If the top symbol is a `Ref`, replace it with that rule's body. If it is a `Chars`, match the next character and pop it. When rules have multiple alternatives, the matcher cannot know which branch the upcoming text will take, so it maintains a *set* of stacks, one per active reading. A character is accepted if at least one stack accepts it, and stacks that fail are discarded. This breadth-first simulation of a nondeterministic pushdown automaton is how llama.cpp matches GBNF.

**Why stacks fail on left recursion.** Consider expanding `expr ::= expr "-" term | term` at the top of a stack. The first alternative pushes `expr` back on top with `"-" term` below it. Expanding `expr` again produces two copies of `"-" term` below it, and so on forever. Because every stack is strictly longer than the last, checking whether a stack has been seen before never stops the loop. A left-recursive grammar requires infinitely many stacks before reading a single character.

What remains finite is the set of *positions inside rules*. An **Earley item** `(rule, alt, dot, origin)` records how far one alternative (`alt`) of `rule` has progressed (`dot`) and the character position where it began (`origin`), without storing the caller stack above it. When an item finishes, its waiting parents are recovered from column `origin`. They are the items in column `origin` whose dot sits before `Ref(rule)`. Multiple parents share a single child item, and a left-recursive rule simply appears among its own parents. After $n$ characters, a column (the set of active items at position $n$) has bounded size, so expanding active items always terminates.

At each character, three operations advance the parser:

- **Scan:** For every item whose dot stands before a `Chars` matching the character, add the item with `dot + 1` to the next column.
- **Predict:** For every item whose dot stands before `Ref(target)`, add `(target, alt, 0, position)` for each alternative `alt` of `target` to the current column.
- **Complete:** For every item whose dot is at the end of its body, find every item in column `origin` whose dot stands before `Ref(rule)` and add it with `dot + 1` to the current column.

Repeating predict and complete until the column stops growing is called **closure**. One subtlety arises with nullable rules. A rule that matches the empty string completes in the same column where it is predicted, while other parents may still be entering that column. Any parent that arrives after the empty rule has already been processed would never be advanced. The Aycock-Horspool fix is to immediately advance `(rule, alt, dot + 1, origin)` during prediction whenever `target` is in `grammar.nullable`.

Because every rule in a kit grammar can derive at least one string, a prefix is viable if and only if its last column is non-empty. The full text is accepted if and only if the last column contains a finished `root` item with `origin == 0`.

**From characters to tokens.** A token is allowed if all of its bytes can be fed in sequence. Running closure after every completed character allows tokens to open and close arbitrary rules mid-token. Two details remain:

- **UTF-8 bytes.** Tokens are byte sequences while grammars match Unicode code points, so the state also stores up to three pending bytes. While a character is incomplete, the Earley columns do not advance. Instead, the pending bytes define a range of possible code points (the payload bits read so far, with unread bits ranging from all 0s to all 1s), clipped to the valid range for that UTF-8 byte length with surrogates excluded. The partial character is viable if at least one item in the last column expects a `Chars` overlapping that range. Once the final byte arrives, scan the resulting code point normally.
- **Shared token prefixes.** Testing each token independently parses `t` four times for `t`, `tr`, `tru`, and `true`. Storing the vocabulary in a byte trie and walking it depth-first carries the matcher state along each edge. A failing byte prunes the entire subtree below it, and every token attached to a reachable node is allowed.

## Reference solution (Python)

```python
class GrammarMatcher:
    def __init__(self, grammar, vocab):
        self.rules = grammar.rules
        self.nullable = grammar.nullable
        self.root = _Node()  # the vocabulary trie
        for token in range(vocab.size):
            piece = vocab.bytes(token)
            if not piece:
                continue
            node = self.root
            for byte in piece:
                node = node.children.setdefault(byte, _Node())
            node.tokens.append(token)

    def start(self):
        first = {("root", alt, 0, 0) for alt in range(len(self.rules["root"]))}
        return (self._close(first, ()),), b""

    def allowed(self, state):
        found = []

        def walk(node, here):
            for byte, child in node.children.items():
                after = self._feed(here, byte)
                if after is not None:
                    found.extend(child.tokens)
                    walk(child, after)

        walk(self.root, state)
        return sorted(found)

    def _feed(self, state, byte):
        columns, pending = state
        pending += bytes([byte])
        points = _code_points(pending)  # (low, high) or None
        if points is None:
            return None
        low, high = points
        expected = [self.rules[n][a][d] for n, a, d, _ in columns[-1] if d < len(self.rules[n][a])]
        if low < high or len(pending) < len(chr(low).encode()):
            if any(isinstance(s, Chars) and s.overlaps(low, high) for s in expected):
                return columns, pending
            return None
        scanned = {
            (n, a, d + 1, o)
            for n, a, d, o in columns[-1]
            if d < len(self.rules[n][a])
            and isinstance(self.rules[n][a][d], Chars)
            and self.rules[n][a][d].matches(low)
        }
        if not scanned:
            return None
        return (*columns, self._close(scanned, columns)), b""

    def _close(self, items, columns):
        position = len(columns)
        current, todo = set(items), list(items)
        while todo:
            name, alt, dot, origin = todo.pop()
            body = self.rules[name][alt]
            new = []
            if dot == len(body):  # complete
                waiting = columns[origin] if origin < position else tuple(current)
                for p_name, p_alt, p_dot, p_origin in waiting:
                    p_body = self.rules[p_name][p_alt]
                    if p_dot < len(p_body) and p_body[p_dot] == Ref(name):
                        new.append((p_name, p_alt, p_dot + 1, p_origin))
            elif isinstance(body[dot], Ref):  # predict
                target = body[dot].name
                new.extend((target, a, 0, position) for a in range(len(self.rules[target])))
                if target in self.nullable:
                    new.append((name, alt, dot + 1, origin))
            for item in new:
                if item not in current:
                    current.add(item)
                    todo.append(item)
        return frozenset(current)
```

`advance` feeds the token's bytes one by one and raises `ValueError` if any `_feed` call returns `None`. `is_accepting` checks that `pending` is empty and that the last column contains a completed `root` item with `origin == 0`. `_code_points` implements the UTF-8 table from the statement. The lead byte sets the expected length and top payload bits, each continuation byte adds 6 bits, and the unread bits determine `[low, high]`, which is then clipped to the valid non-surrogate code points of that length. Storing the state as a tuple of frozensets plus `bytes` makes it immutable and hashable.

## Common mistakes

- **Keeping only one stack.** Committing to the first alternative that matches the next character works on LL(1) grammars like the list example, but fails as soon as two alternatives share a prefix or a repetition can either stop or continue.
- **Recursively expanding the top of a stack.** This works for non-left-recursive grammars and loops forever on `expr ::= expr "-" term`. Whole-stack cycle detection does not help because the stacks grow without bound.
- **Skipping nullable rules during prediction.** This classic Earley bug only surfaces when the same nullable rule can match twice at one position (such as `ws items ws` when `items` is empty), so it often slips past small tests.
- **Closing the item set only between tokens.** Scanning all characters of a token before running predict and complete accepts tokens inside a single literal (like `true`) but rejects tokens that cross rule boundaries (like `, ` or `{"`).
- **Decoding tokens before matching.** Calling `piece.decode()` raises an error on tokens that hold half a UTF-8 character, making split multi-byte characters unreachable.
- **Accepting any partial UTF-8 sequence.** Allowing pending bytes without checking `Chars.overlaps`, or computing the code-point range without clipping to valid lengths (so `F0` appears to overlap ASCII starting at 0), lets invalid byte sequences through.
- **Including special tokens in the trie.** Feeding zero bytes never fails, so tokens with `vocab.bytes(t) == b""` must be skipped when building the trie.
- **Accepting on any finished `root` item.** When `root` is recursive (`value ::= ... | root`), an inner `root` finishes before the outer input is complete. Acceptance requires `origin == 0` and no pending bytes.
- **Calling the oracle.** Using `grammar.accepts` or `grammar.accepts_prefix` is forbidden by the harness and re-parses the full text for every token at every step.

## Complexity

For a grammar of total size $G$ after $n$ characters, a column contains at most $O(G \cdot n)$ items and closing it takes up to $O(G^2 n^2)$ time in the worst case. For the unambiguous grammars used in structured output, columns stay small and per-character work is near constant. Each call to `allowed` runs `_feed` once per visited trie edge, bounded by the total number of bytes in the vocabulary and much smaller in practice because most first bytes fail immediately. The reference state retains every column for completion; production engines prune or share unreachable columns.

## In the wild

- **llama.cpp** parses GBNF into the same rule structure as `Grammar.rules` and matches using a set of stacks while carrying partial UTF-8 bytes in the state. Left recursion is a known hazard of stack-based matching, so llama.cpp checks for it when loading a grammar.
- **XGrammar** ("XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models", arXiv 2411.15100) uses a stack-based pushdown automaton with persistent stacks that share tails across branches. It splits the vocabulary into context-independent tokens (precomputed per rule position) and a small set of context-dependent tokens checked at runtime. **XGrammar-2** ("XGrammar-2: Dynamic and Efficient Structured Generation Engine for Agentic LLMs", arXiv 2601.04426) extends this to grammars that change dynamically during generation.
- **llguidance** combines an Earley parser with a regex-based lexer and walks a vocabulary trie to compute the token mask, using the same two core ideas as this solution.
- Across all of these engines, the resulting mask feeds into the decoding loop of `constrained-greedy-decode`. Precomputing masks per automaton state (as in `dfa-token-index` for regular languages) is the simpler version of XGrammar's context-independent token cache.
