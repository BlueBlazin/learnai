## The idea

A schema of this subset describes a regular language, as long as nesting is finite:
every construct is a concatenation, a choice, or a bounded or unbounded repetition
of the constructs inside it. So the conversion is a recursive function that returns
a pattern for each schema and pastes the patterns of sub-schemas into it.

Most of it is transcription.

- **Scalars.** `null`, `(true|false)`, `-?(0|[1-9][0-9]*)`, and for numbers the
  integer followed by `(\.[0-9]+)?([eE][+\-]?[0-9]+)?`. The integer's shape
  (`0`, or a non-zero digit and more) is what excludes leading zeros.
- **Strings.** One item is `[^"\\\x00-\x1f]`, a two-character escape, or `\u` and four
  hex digits; the string is the item repeated between quotes, and a length bound
  becomes the repetition's `{min,max}`. The bound counts items, so `\n` is one.
- **Enums.** Each value's `json.dumps` text, escaped, in an alternation.
- **Arrays.** A comma-separated list is "one item, then (comma, item) repeated":
  `item(,item){min-1,max-1}`. The commas are tied to the items after the first, so
  none can dangle, and the bounds shift by one because the first item is outside
  the repetition. The empty array is the whole body made optional, when `minItems`
  is 0.

**Objects** are the one place that needs an idea. With members in a fixed order,
some of them optional, the documents are all subsequences of the member list that
contain every required member, joined by commas. "Optional" suggests `(member,)?`
or `(,member)?`, and each is right only in half of the cases: whether a member has
a comma before it depends on whether anything precedes it, which is not a property
of that member.

The fix is to make that fact part of the recursion. Let $M_j$ be member $j$'s
pattern (`"name":` and the value). Define two patterns for "the members from
position $i$ on":

$$
\text{rest}(i) = \prod_{j \ge i} \begin{cases} \texttt{,}M_j & j \text{ required} \\ \texttt{(,}M_j\texttt{)?} & j \text{ optional} \end{cases}
$$

for the case that a member has already been written (every further member brings
its own comma), and

$$
\text{first}(i) = \begin{cases}
\varepsilon & i \text{ past the end} \\
M_i\ \text{rest}(i+1) & i \text{ required} \\
\texttt{(}M_i\ \text{rest}(i+1)\ \texttt{|}\ \text{first}(i+1)\texttt{)} & i \text{ optional}
\end{cases}
$$

for the case that nothing has been written. The object is `\{` first(0) `\}`.
Correctness is by induction on $i$: rest($i$) matches exactly the valid
continuations after a written member, and first($i$) exactly the valid lists that
start at $i$ with no member before them. Once a required member is reached,
first stops branching, so the pattern grows only with the run of optional
properties at the start: linear when the first property is required, quadratic in
that run otherwise (twelve leading optional properties give twelve nested
alternatives, not $2^{12}$).

This two-state recursion is a small automaton written as a regex: the state is
"has a member been written". Formats with more context than that (properties in
any order, recursive schemas) do not fit a regular expression of reasonable size,
which is where grammar-based constraints take over.

Two rules about the output are as important as the language. **Escape the data**:
names and enum values go through a function that puts a backslash before every
metacharacter. And **write for an automaton**: no anchors, no lookaround, digits
as `[0-9]`. The pattern's consumer is a DFA compiler over bytes, and `\d` in a
Python `str` pattern is a Unicode class.

## Reference solution (Python)

```python
INTEGER = r"-?(0|[1-9][0-9]*)"
NUMBER = INTEGER + r"(\.[0-9]+)?([eE][+\-]?[0-9]+)?"
STRING_ITEM = r'([^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})'
METACHARACTERS = set(".^$*+?{}[]()|\\")


def literal(text):
    return "".join("\\" + char if char in METACHARACTERS else char for char in text)


def string(schema):
    low = schema.get("minLength", 0)
    high = schema.get("maxLength")
    if high is None:
        count = "*" if low == 0 else f"{{{low},}}"
    else:
        count = f"{{{low},{high}}}"
    return f'"{STRING_ITEM}{count}"'


def array(schema):
    item = schema_to_regex(schema["items"])
    low = schema.get("minItems", 0)
    high = schema.get("maxItems")
    if high == 0:
        return r"\[\]"
    more = max(low, 1) - 1
    count = f"{{{more},}}" if high is None else f"{{{more},{high - 1}}}"
    body = f"{item}(,{item}){count}"
    return rf"\[{body}\]" if low >= 1 else rf"\[({body})?\]"


def obj(schema):
    required = set(schema.get("required", ()))
    members = [
        (literal(json.dumps(name)) + ":" + schema_to_regex(sub), name in required)
        for name, sub in schema.get("properties", {}).items()
    ]

    def rest(i):
        return "".join(f",{m}" if needed else f"(,{m})?" for m, needed in members[i:])

    def first(i):
        if i == len(members):
            return ""
        member, needed = members[i]
        present = member + rest(i + 1)
        return present if needed else f"({present}|{first(i + 1)})"

    return r"\{" + first(0) + r"\}"


def schema_to_regex(schema):
    if "enum" in schema:
        return "(" + "|".join(literal(json.dumps(value)) for value in schema["enum"]) + ")"
    kind = schema.get("type")
    if kind == "null":
        return "null"
    if kind == "boolean":
        return "(true|false)"
    if kind == "integer":
        return INTEGER
    if kind == "number":
        return NUMBER
    if kind == "string":
        return string(schema)
    if kind == "array":
        return array(schema)
    if kind == "object":
        return obj(schema)
    raise ValueError(f"unsupported schema type: {kind!r}")
```

Every function returns a pattern that is safe to concatenate: alternations are
always inside a group, and the scalar patterns have none at the top level.

## Common mistakes

- **A comma glued to an optional member.** `(member,)?` leaves `{"a":1,}` when the
  last member is absent; `(,member)?` leaves `{,"b":2}` when the first is. Both
  also reject valid documents. This is the mistake the object rule exists for.
- **`required` ignored**, so that `{}` matches a schema that demands a field. The
  generation then "succeeds" with an object the caller cannot use.
- **Strings as `"[^"]*"`.** An escaped quote ends the string early, so a value with
  a quotation mark in it cannot be generated at all, and raw newlines and lone
  backslashes, which no JSON parser accepts, can.
- **Control characters allowed.** `[^"\\]` forgets U+0000 to U+001F, and a literal
  newline inside a string, which no JSON parser accepts, becomes a legal output.
- **A length bound on the wrong unit**, for example applied to a character class
  without the escape alternatives, so that a bounded string cannot contain `\n`.
- **`maxItems` off by one.** The repetition counts the items after the first:
  `{min-1,max-1}`, and `[]` is separate.
- **No empty array.** `\[item(,item)*\]` requires one item.
- **Names and enum values not escaped.** `a.b` then also matches `axb`, `c++` is a
  different pattern or an invalid one, and a `|` in a value splits the alternation.
- **`str(value)` for enum values.** `True` and `None` are not JSON, and strings
  need their quotes.
- **Leading zeros** (`-?[0-9]+`) and **`\d`**, which without `re.ASCII` matches
  digits of other scripts.
- **An exponent that requires a sign.** JSON allows `1e5`.
- **Whitespace "for readability".** An optional space after separators is a
  different, larger language: it matches documents that are not compact, and it
  gives the model a way to pad the output forever if the whitespace is unbounded.
- **Anchors or lookaround in the pattern.** They work with `re.fullmatch` and are
  rejected by the automaton compiler that consumes the pattern.

## Complexity

The pattern is linear in the size of the schema, with three multipliers: a nested
schema's pattern is repeated wherever it is pasted (twice per array level, since
`item` appears twice), a run of $k$ optional properties at the start of an object
costs $O(k^2)$ members, and bounds are cheap in the pattern (`{0,100}`) but expand
in the automaton: a string with `maxLength` 100 is a hundred copies of the item's
states. The last one is the cost that matters in practice.

## In the wild

This is the approach of Outlines, described in "Efficient Guided Generation for
Large Language Models" (arXiv 2307.09702) and implemented in `outlines-core`
(`src/json_schema`, read on its main branch). What is the same: the recursion over
the schema, properties in declaration order, string and number patterns of the
shape used here, and the split of the object rule into two cases. `outlines-core`
writes the cases differently: with at least one required property, the members
before the last required one carry a trailing comma and those after it a leading
one; with none required, it emits one alternative per choice of the last present
property. What differs, as of that reading:

- **Whitespace.** `outlines-core` inserts a configurable whitespace pattern between
  tokens, by default `[ ]?` (at most one space). This problem is compact, which is
  the pattern with that option set to the empty string.
- **Numbers.** Its number pattern is `((-)?(0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?`:
  an exponent must carry a sign. This problem follows RFC 8259, where the sign is
  optional.
- **Strings.** Its string item is `[^"\\\x00-\x1F\x7F-\x9F]|\\["\\/bfnrt]`: no `\u`
  escapes, and U+007F to U+009F excluded. This problem follows RFC 8259, which
  allows both.

Outlines supports more of JSON Schema than this subset; none of that is attempted
here.
