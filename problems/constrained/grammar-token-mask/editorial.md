## The idea

**From one stack to many.** A grammar without alternatives can be matched with one
stack: the symbols still to be read, next one on top. A rule reference on top is
replaced by the rule's body; a character set on top is compared with the next
character and popped. With alternatives a reference has several replacements, and
the matcher cannot know which one the text will follow, so it keeps a *set* of
stacks, one per reading that is still possible. A character is accepted if some
stack accepts it, and the stacks that do not are dropped. This is a nondeterministic
pushdown automaton simulated breadth-first, and it is how llama.cpp matches GBNF.

**Why the stacks have to go.** Expand `expr ::= expr "-" term | term` on top of a
stack and the first alternative puts `expr` on top again, with `"-" term` beneath
it; expand that and there are two copies beneath, and so on. Each stack is a
different, longer one, so "have I seen this stack before" never fires. The set of
stacks that a left-recursive grammar needs before reading a single character is
infinite.

What is finite is the set of *positions inside rules*. An **Earley item**
`(rule, alternative, dot, origin)` records how far one alternative has got and at
which character it started, and nothing about what is above it. The stack is
recovered when it is needed: when an item finishes, the items that were waiting for
that rule are exactly the items in column `origin` whose dot stands before a
reference to it. Many parents share one child item, and a left-recursive rule simply
has itself among its parents. A column, the set of items after a given number of
characters, has at most (rules' total length) × (positions) elements, so everything
terminates.

Per character:

- **scan**: items whose dot is before a `Chars` that matches the character move, with
  the dot advanced, into the next column;
- **predict**: for an item whose dot is before a rule, add that rule's alternatives
  with dot 0 and origin "here";
- **complete**: for an item whose dot is at the end, advance its parents (found in
  column `origin`) into the current column.

Predict and complete are repeated until nothing new appears. One subtlety: a rule
that matches nothing completes in the same column it was predicted in, where its
parents are still arriving. An item that arrives after the empty rule has already
been completed would wait forever. The fix (Aycock and Horspool's) is to advance
over a nullable rule at prediction time.

A text is a viable prefix exactly when its last column is not empty, because every
rule in a kit grammar can finish. It is a member of the language when the last
column holds a finished `root` item with origin 0.

**From characters to tokens.** A token is allowed when all its bytes can be fed in
turn. Nothing about rule boundaries enters: closure runs after every character, so a
token may finish one rule and begin the next. Two things do need care.

*Bytes.* Tokens carry bytes, the grammar reads code points. The state therefore
includes up to three pending bytes. While a character is incomplete the column does
not move; the matcher only asks whether continuing is possible: the pending bytes
determine a range of code points (the payload bits read so far, with the unread
bits free), clipped to the characters that really have that encoding length, and
the partial character is viable if some item's next symbol is a `Chars` that
overlaps the range. When the last byte arrives, the code point is scanned as usual.

*Shared prefixes.* Checking each token separately parses `t` four times for `t`,
`tr`, `tru`, `true`. Put the vocabulary into a trie over bytes and walk it once,
depth-first, carrying the matcher state along each edge. A byte that fails prunes
the whole subtree; a node's tokens are allowed when the walk reaches it.

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

`advance` feeds the token's bytes and raises `ValueError` on a `None`;
`is_accepting` is "no pending bytes and a finished `root` item with origin 0 in the
last column". `_code_points` implements the table in the statement: the lead byte
gives the length and the first payload bits, continuation bytes add six bits each,
the unread bits give the range, and the range is clipped to the code points of that
length without surrogates. The state is a tuple of frozensets and a `bytes`, so it
is hashable and old states stay valid.

## Common mistakes

- **One stack.** Committing to the first alternative that fits the next character
  is right for grammars where one character always decides (the list example) and
  wrong as soon as two alternatives begin alike or a repetition may stop or go on.
- **Recursive expansion of the top of the stack.** Correct for every grammar
  without left recursion, and unbounded recursion on `expr ::= expr "-" term`.
  Cycle detection on whole stacks does not help, because the stacks keep growing.
- **Forgetting empty rules at prediction.** The classic Earley bug. It needs the
  same nullable rule twice at one position (`ws items ws` with empty `items`) to
  show, which is why it survives small tests.
- **Closing the item set only between tokens.** Scanning a token's characters
  without predict and complete in between accepts tokens that stay inside one
  literal (`true`) and rejects those that cross into another rule (`, `, `{"`).
- **Decoding tokens before matching.** `piece.decode()` fails on a token that is
  half a character, and treating that as "not allowed" makes every character that
  the tokenizer splits unreachable.
- **Allowing any partial character.** The opposite error: accepting pending bytes
  because they begin *some* character, without asking the grammar for an overlap,
  or computing the range from the payload bits alone, which lets `F0` (whose
  unclipped range starts at 0) stand for an ASCII character.
- **Special tokens.** A walk over zero bytes never fails; tokens without bytes must
  be left out of the trie.
- **`is_accepting` on any finished `root`.** With `value ::= ... | root` an inner
  `root` finishes long before the text is complete. The origin must be 0, and no
  bytes may be pending.
- **Calling the oracle.** It gives the right answers and re-parses the whole text
  for every token of the vocabulary at every step.

## Complexity

A column has $O(G \cdot n)$ items for a grammar of total size $G$ after $n$
characters, and closing it costs up to $O(G^2 n^2)$ in the worst case, though for
the unambiguous grammars of structured output columns stay small and the work per
character is near constant. `allowed` performs one feed per trie edge reached, at
most the total number of bytes in the vocabulary and usually far fewer, since most
first bytes fail. The state as written keeps every column, which is what completion
needs; real engines drop or share what can no longer be reached.

## In the wild

- **llama.cpp** parses GBNF into the rule shape `Grammar.rules` mirrors and matches
  with a set of stacks, carrying a partial UTF-8 sequence in the grammar state
  between tokens. Left recursion is a known hazard of that design; its GBNF
  documentation (`grammars/README.md`) does not mention it, and the project added a
  check for it when grammars are loaded.
- **XGrammar** ("XGrammar: Flexible and Efficient Structured Generation Engine for
  Large Language Models", arXiv 2411.15100) keeps the stack-based automaton and makes
  it fast: stacks are stored persistently so that branches share their tails, and
  the vocabulary is split into tokens whose validity depends only on the position
  inside the current rule, which are precomputed, and the few context-dependent
  ones that are checked at run time. Its successor's report ("XGrammar-2: Dynamic
  and Efficient Structured Generation Engine for Agentic LLMs", arXiv 2601.04426)
  extends this to formats that change during generation.
- **llguidance** uses an Earley parser for the grammar on top of a lexer built from
  regular expressions, and walks a trie of the vocabulary to compute the mask, the
  two ideas of this solution.
- Whatever the algorithm, the mask is consumed by the loop of
  `constrained-greedy-decode`, and precomputing it per automaton state, as
  `dfa-token-index` does for regular languages, is what the context-independent
  token cache generalizes.
