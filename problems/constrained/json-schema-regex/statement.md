Turn a JSON Schema into a regular expression that matches exactly the JSON documents
the schema allows. This is the first stage of schema-constrained generation: the
expression is compiled to an automaton, and the automaton masks tokens. JSON's grammar
is easy; the work is in what a regular expression cannot say directly, above all
properties that may be absent without leaving a comma behind.

![This problem is Schema to regex, the first stage of compiling a schema.](assets/map.svg)

## Specification

```python
def schema_to_regex(schema: dict) -> str: ...
```

`schema` is a JSON Schema as a Python dict, restricted to the subset below.
`schema_to_regex` returns a pattern `p` such that, for every string `d`,
`re.fullmatch(p, d)` (no flags) matches exactly when `d` is a **compact** JSON
document that conforms to the schema, as defined here.

**Compact** means no whitespace anywhere outside strings: no space after `:` or `,`,
none inside brackets, none before or after the document. It is the format of
`json.dumps(value, separators=(",", ":"))`.

### The schema subset and its documents

| schema | documents |
|---|---|
| `{"type": "null"}` | `null` |
| `{"type": "boolean"}` | `true` or `false` |
| `{"type": "integer"}` | an optional `-`, then `0` or a digit `1`–`9` followed by any digits `0`–`9` |
| `{"type": "number"}` | an integer as above, then optionally `.` and one or more digits, then optionally `e` or `E`, an optional `+` or `-`, and one or more digits |
| `{"type": "string"}` | `"`, any number of string items, `"` |
| `{"enum": [v1, v2, ...]}` | exactly `json.dumps(v)` for one of the listed values |
| `{"type": "array", "items": S}` | `[`, values of `S` separated by `,`, `]`; possibly `[]` |
| `{"type": "object", "properties": {...}, "required": [...]}` | `{`, members separated by `,`, `}` |

- **Digits** are the ASCII digits `0` to `9` only. `-0` is an integer; `01`, `+1`,
  `1.`, `.5`, and `1e` are not numbers.
- **String items.** A string item is one of
  - one character other than `"`, `\`, and the control characters U+0000 to U+001F
    (every other character, ASCII or not, including U+007F, stands for itself);
  - `\` followed by one of `"`, `\`, `/`, `b`, `f`, `n`, `r`, `t`;
  - `\u` followed by exactly four hexadecimal digits (`0`–`9`, `a`–`f`, `A`–`F`).
- **`minLength`, `maxLength`** (optional keys of a string schema, non-negative ints,
  `minLength <= maxLength`) bound the number of **string items**: an escape sequence
  counts as one item. Without them any number of items is allowed.
- **`enum`.** The values are strings of printable ASCII characters, ints, `True`,
  `False`, or `None`. A document matches if it is, character for character, the
  `json.dumps` form of one of them: `"red"` with its quotes, `3`, `true`, `null`; a
  `"` or `\` inside a string value appears as `\"` or `\\`. If a schema has `enum`,
  its other keys are ignored.
- **Arrays.** `items` is always present and is a schema of the subset. `minItems`
  (default 0) and `maxItems` (default: no limit) bound the number of elements,
  `minItems <= maxItems`.
- **Objects.** `properties` (default: none) maps names to schemas; `required`
  (default: none) lists the names that must be present. A member is written
  `"name":value`, the name as `json.dumps(name)` writes it (names are printable
  ASCII without `"` and `\`) and the value a document of the property's schema.
  - Members appear in the order in which `properties` declares them (the dict's
    order).
  - Every required property appears exactly once. Every other property appears at
    most once.
  - No other member is allowed.
  - Members are separated by single commas: there is no comma after `{`, none before
    `}`, and never two in a row, whichever optional properties are absent. An object
    with no member is `{}`.
- Schemas nest to any depth through `items` and `properties`, and any schema of the
  subset may be the top level (a document can be a bare number).
- A schema that has no `enum` and whose `type` is not one of the seven strings above
  (a missing `type`, a list of types, `"date"`) raises `ValueError`, also when it is
  nested inside a supported schema. Tests use no other key than those named here.

### The pattern

The pattern is written for an automaton, not for a backtracking engine. It must be
valid for Python's `re` and must stay inside the regex subset of
[`regex-to-dfa`](../regex-to-dfa/statement.md), which is what the kit's
`CharDFA.from_regex` compiles: literals, escapes, classes, `.`, `|`, groups, and the
quantifiers `* + ? {m} {m,} {m,n}` with bounds up to 100. In particular: no anchors
(`^`, `$`; the whole document must match anyway), no lookaround, no backreferences,
no lazy quantifiers, no `\uXXXX` or `\b` escapes in the pattern.

Because the match is `re.fullmatch` without flags, `\d` would also match non-ASCII
decimal digits such as `٣`; such a document is not JSON.

Property names and enum values are data, not syntax: a name like `a.b` or a value
like `c++` must match only itself.

## Example

```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string"},
    },
    "required": ["name", "age"],
}
```

| document | conforms | why |
|---|---|---|
| `{"name":"Ada","age":36}` | yes | `email` is optional |
| `{"name":"Ada","age":36,"email":"ada@example.com"}` | yes | |
| `{"name": "Ada", "age": 36}` | no | not compact |
| `{"age":36,"name":"Ada"}` | no | members out of order |
| `{"name":"Ada"}` | no | `age` is required |
| `{"name":"Ada","age":36,}` | no | a comma without a member |
| `{"name":"Ada","age":036}` | no | leading zero |
| `{"name":"Ada","age":36,"x":1}` | no | `x` is not a property |

One pattern that does this, with `STR` standing for the string pattern
`"([^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*"`:

```
\{"name":STR,"age":-?(0|[1-9][0-9]*)(,"email":STR)?\}
```

The optional member carries its own comma inside the `(...)?`. That works here
because a required member always precedes it. With all three properties optional
there is no such anchor: `{"email":"a"}` has no comma, `{"name":"n","email":"a"}`
has one, and the pattern has to say which member comes first.

Any pattern with the same language is accepted; the tests never look at its text.

## Constraints

- At most 12 properties per object and 4 levels of nesting; `maxLength` and
  `maxItems` at most 100.
- The verdict on every document comes from a direct matcher of the rules above, not
  from another regular expression. Tests use hand-written documents, generated
  instances of the schema (which must match), and corrupted copies of them.
- The pattern must compile with `re`, and, where the tests check it (schemas with
  small bounds), with `CharDFA.from_regex`.
- The standard library is allowed, including `json` and `re.escape`.

## Why this matters

In engines that follow the Outlines approach, "JSON mode with a schema" is this
function followed by `regex-to-dfa` and `dfa-token-index`. The model can then only emit what the pattern
allows, which cuts both ways: a pattern that is too loose lets through documents the
caller's parser rejects (a trailing comma, `01`, a raw newline in a string), and a
pattern that is too strict forbids documents the model needs (an escaped quote, an
exponent without a sign) and forces it into something else that happens to match.
Neither produces an error at generation time. The object rule is where conversions
go wrong most often, because "optional" is not local in a comma-separated list:
whether a member needs a comma depends on whether anything came before it.
