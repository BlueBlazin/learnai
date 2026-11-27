Build the pattern recursively with one helper per schema type, pasting the compiled patterns of nested sub-schemas into the parent pattern. Every type except `object` translates directly from its grammar. For `object`, write out every valid document for a schema with three optional properties `a`, `b`, and `c` before writing any regex. What determines whether a member has a comma in front of it?

---

A member has a leading comma if and only if at least one member was written before it. That gives two cases for matching the suffix of members starting at index `i`:

- **At least one member was already written.** Every remaining member carries its own leading comma: `,member` if required, or `(,member)?` if optional. Concatenate them in order.
- **No member has been written yet.** If member `i` is required, emit it without a comma and follow it with the first case for `i + 1`. If member `i` is optional, form an alternation between emitting it (followed by the first case for `i + 1`) and skipping it (recursing in this second case at `i + 1`).

Arrays use the same idea in a simpler form: match one item followed by repeated `(,item)` pairs so the number of commas is always one less than the number of items. Handle the empty array `[]` separately when `minItems` is `0`.

---

```
rest(i)  = concat over j >= i of:  ",M_j"  if required else  "(,M_j)?"
first(i) = ""                              if i is past the last member
         = M_i rest(i+1)                   if member i is required
         = "(" M_i rest(i+1) "|" first(i+1) ")"   if member i is optional
object   = "\{" first(0) "\}"

array    = "\[" item "(," item "){lo-1,hi-1}" "\]"        with minItems >= 1
         = "\[(" item "(," item "){0,hi-1})?" "\]"        with minItems = 0
string   = '"' ITEM '{lo,hi}"'   ITEM = ([^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})
```

Here `M_j` is the escaped `json.dumps(name)`, followed by `:`, followed by the property's compiled schema pattern, and `lo` and `hi` are the lower and upper bounds (`minItems`/`maxItems` or `minLength`/`maxLength`, omitting `hi` when unbounded). Escape every regex metacharacter (`. ^ $ * + ? { } [ ] ( ) | \`) in property names and enum values, write digits as `[0-9]` instead of `\d`, and wrap every alternation in parentheses so it cannot capture adjacent characters.
