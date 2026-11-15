Start with one character at a time and forget tokens. If the grammar had no
alternatives, a single stack of "symbols still to match" would do. With alternatives
you need a set of such stacks. Now write down the stacks for
`expr ::= expr "-" term | term` before any character has been read. How many are
there? What do they have in common, and what would you have to remember about a
rule instead of the whole stack above it?

---

Replace the stacks by Earley items. An item is `(rule, alternative, dot, origin)`:
this alternative has matched its first `dot` symbols, and it began at character
number `origin`. Keep one set of items per character position (a "column"). Three
operations close a column: *predict* (the symbol after the dot is a rule: add its
alternatives with dot 0 and the current position as origin), *complete* (the dot is
at the end: every item in column `origin` whose dot stands before this rule advances
by one, into the current column), and, for the next character, *scan* (every item
whose dot stands before a `Chars` that matches moves into the next column). Because
a column is a set, a left-recursive rule predicts itself once and stops; because an
item names its origin instead of carrying its parents, common parents are shared. A
state is the tuple of columns so far plus the bytes of an unfinished character.

---

Three details finish it.

Empty rules: when you predict a rule that is in `grammar.nullable`, also add the
predicting item with its dot moved past the rule. Completion alone misses items that
arrive in the column after the empty rule was completed there.

Bytes: collect bytes until they form a character (see the table in the statement),
then scan its code point. For an incomplete sequence compute the range of code
points that start with those bytes: with `k` of `n` bytes read and payload `v`,
`low = v << 6*(n-k)` and `high = low | ((1 << 6*(n-k)) - 1)`, clipped to the code
points that take `n` bytes and with the surrogates removed. The partial character is
viable if the range is not empty and some item's next symbol is a `Chars` with
`overlaps(low, high)`. Leave the column untouched until the character is complete.

Tokens: `allowed` feeds each token's bytes one by one and keeps the token if no byte
fails. Put the vocabulary in a trie keyed by bytes and walk it depth-first, carrying
the matcher state down each edge: `tr`, `tru`, and `true` then share the work for
`t` and `r`, and a dead byte prunes every token below it. `is_accepting`: no pending
bytes, and the last column has a finished `root` item with origin 0.
