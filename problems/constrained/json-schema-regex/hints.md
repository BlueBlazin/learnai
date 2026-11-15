Build the pattern recursively: one function per schema type, each returning the
pattern of its documents, with the patterns of nested schemas pasted in. Every type
but one is a direct transcription of its grammar. For the object, write down every
document of a schema with three optional properties `a`, `b`, `c` before writing
any pattern. What decides whether a member has a comma in front of it?

---

A member has a comma in front of it exactly when some member was written before it.
So there are two situations, and each has a simple pattern for "the members from
position `i` on":

- **Something was already written.** Every remaining member brings its own leading
  comma: `,member` if it is required, `(,member)?` if it is optional. Concatenate.
- **Nothing was written yet.** If member `i` is required, it comes without a comma,
  followed by the first situation for the members after it. If it is optional,
  either it is present (the same, in a group) or it is absent and the question
  repeats for member `i + 1`: an alternation of the two.

An array is the same idea in its simplest form: one item, then `(,item)` repeated,
so the number of commas is always one less than the number of items; the empty
array is a separate, optional case.

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

where `M_j` is the escaped `json.dumps(name)`, a colon, and the pattern of the
property's schema. Escape every regex metacharacter in names and enum values
(`. ^ $ * + ? { } [ ] ( ) | \`), write digits as `[0-9]`, and wrap every
alternation in a group so that it cannot capture its neighbors.
