Start by matching one character at a time and ignore tokens for now. If a grammar had no alternatives, a single stack of remaining symbols would suffice. With alternatives, you need a set of stacks. Now write down the stacks for `expr ::= expr "-" term | term` before reading any character. How many stacks exist, what do they share, and what can you store about a rule instead of its entire caller stack?

---

Replace stacks with Earley items of the form `(rule, alternative, dot, origin)`. An item means that `alternative` of `rule` has matched its first `dot` symbols and started at character position `origin`. Store one set of items per character position (a *column*).

Three operations drive the parser:
- **Predict:** if the symbol at `dot` is a `Ref`, add each alternative of that rule with `dot = 0` and `origin = current_position`.
- **Complete:** if `dot` is at the end of the alternative, advance every item in column `origin` whose dot stands before this rule and add it to the current column.
- **Scan:** for the next character, advance every item whose dot stands before a matching `Chars` into the next column.

Because each column is a set, a left-recursive rule predicts itself once and stops. Because items reference `origin` instead of copying caller stacks, shared parents are merged automatically. A state consists of the tuple of columns so far plus any pending UTF-8 bytes.

---

Three final details complete the matcher:

1. **Nullable rules:** when predicting a rule listed in `grammar.nullable`, also add the predicting item with its dot advanced by one. Completion alone misses items that enter the column after the empty rule was already completed.
2. **UTF-8 bytes:** buffer bytes until they form a complete character (using the table in the statement), then scan its code point. For an incomplete sequence with `k` of `n` bytes read and accumulated payload bits `v`, compute `low = v << (6 * (n - k))` and `high = low | ((1 << (6 * (n - k))) - 1)`. Clip `[low, high]` to the valid code points for length `n` and exclude surrogates. The partial sequence is viable if the range is non-empty and some item in the last column expects a `Chars` where `overlaps(low, high)` is true. Leave the columns unchanged until the final byte arrives.
3. **Tokens and acceptance:** store the vocabulary in a byte trie (skipping special tokens with `b""`) and walk it depth-first in `allowed`, carrying the matcher state down each edge so `tr`, `tru`, and `true` share prefix work and dead bytes prune whole subtrees. `is_accepting` is true when no bytes are pending and the last column contains a finished `root` item with `origin == 0`.
