## The idea

Every schema in this subset describes a regular language as long as nesting depth is finite. Each construct is built from concatenation, alternation, or repetition of its sub-schemas. That means `schema_to_regex` can be written as a recursive function that compiles each sub-schema into a pattern and embeds it into the parent pattern.

Most schema types translate directly from their grammar:

- **Scalars.** `null` matches `null`, booleans match `(true|false)`, integers match `-?(0|[1-9][0-9]*)`, and numbers append `(\.[0-9]+)?([eE][+\-]?[0-9]+)?` to the integer pattern. Splitting integers into `0` versus `[1-9][0-9]*` is what forbids leading zeros.
- **Strings.** A single string item is `[^"\\\x00-\x1f]`, a two-character escape (`\\["\\/bfnrt]`), or a four-hex-digit Unicode escape (`\\u[0-9a-fA-F]{4}`). Repeating this item inside quotes handles `minLength` and `maxLength` naturally, with each escape counting as one item.
- **Enums.** Serialize each value with `json.dumps`, escape regex metacharacters, and join them with `|` inside a group.
- **Arrays.** A non-empty comma-separated list is one item followed by `(,item)` repeated: `item(,item){min-1,max-1}`. Attaching each comma to the item that follows it prevents dangling commas, and shifting the bounds by one accounts for the first item outside the repetition. When `minItems` is `0`, wrap the non-empty body in `(...)?` so `[]` is also accepted.

**Objects** are the one construct that cannot be translated member by member. With properties in a fixed order, valid objects correspond to all subsequences of the property list that include every required property, joined by commas. Writing an optional member as `(member,)?` or `(,member)?` fails because whether a member needs a comma depends on whether an earlier member was emitted, which is not a local property of that member.

We can track that single bit of state directly in a two-case recursion. Let $M_j$ be the pattern for member $j$ (`"name":` followed by its value pattern). Define two helper patterns for the suffix of members starting at index $i$.

First, $\text{rest}(i)$ matches the remaining members when at least one earlier member has already been emitted, so every subsequent member brings its own leading comma:

$$
\text{rest}(i) = \prod_{j \ge i} \begin{cases} \texttt{,}M_j & j \text{ required} \\ \texttt{(,}M_j\texttt{)?} & j \text{ optional} \end{cases}
$$

Second, $\text{first}(i)$ matches the remaining members when nothing has been emitted yet:

$$
\text{first}(i) = \begin{cases}
\varepsilon & i \text{ past the end} \\
M_i\ \text{rest}(i+1) & i \text{ required} \\
\texttt{(}M_i\ \text{rest}(i+1)\ \texttt{|}\ \text{first}(i+1)\texttt{)} & i \text{ optional}
\end{cases}
$$

The full object pattern is `\{` followed by $\text{first}(0)$ and `\}`. By induction on $i$, $\text{rest}(i)$ matches every valid continuation after an emitted member, and $\text{first}(i)$ matches every valid member list starting at $i$ with no preceding member. Because $\text{first}$ stops branching at the first required property, the pattern only grows with the run of leading optional properties. It is linear when the first property is required, and quadratic in the length of that leading optional run otherwise (twelve leading optional properties produce twelve nested alternatives, not $2^{12}$).

This two-state recursion is a small automaton encoded directly as a regex, where the state is whether a member has been written yet. Formats that require more context, such as arbitrary property order or recursive schemas, cannot be expressed as compact regular expressions, which is where context-free grammar constraints take over.

Two output rules are just as important as the language itself. First, **escape all data**: pass property names and enum values through a helper that backslashes every regex metacharacter. Second, **write for a byte DFA**: avoid anchors, lookarounds, and `\d` (which matches Unicode digits in Python `str` patterns without `re.ASCII`).

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

Every helper returns a pattern that is safe to concatenate directly into a larger expression. All top-level alternations are wrapped in parentheses so they cannot capture neighboring tokens.

## Common mistakes

- **Attaching a comma directly to an optional member.** Writing `(member,)?` allows trailing commas like `{"a":1,}` when the last member is omitted, while `(,member)?` allows leading commas like `{,"b":2}` when the first is omitted. Both also reject valid objects.
- **Ignoring `required`.** Allowing `{}` for a schema with required fields lets generation succeed with an object the caller cannot use.
- **Matching strings with `"[^"]*"`.** This stops early on an escaped quote `\"`, making quotes inside strings impossible to generate, while allowing raw newlines and lone backslashes that JSON parsers reject.
- **Allowing control characters in strings.** Using `[^"\\]` omits the U+0000 to U+001F exclusion and permits literal newlines inside strings.
- **Applying length bounds to the wrong unit.** Putting `{min,max}` on a bare character class without the escape alternatives prevents bounded strings from containing escapes like `\n`.
- **Off-by-one `maxItems`.** Because the first element sits outside the `(,item)` repetition, the quantifier bounds must be shifted down by one (`{min-1,max-1}`), with `[]` handled separately.
- **Forgetting the empty array.** `\[item(,item)*\]` rejects `[]` even when `minItems` is 0.
- **Not escaping property names and enum values.** Unescaped `a.b` matches `axb`, `c++` produces an invalid or wrong regex, and `|` splits an alternation.
- **Using `str(value)` for enum values.** Python's `True` and `None` are not valid JSON, and strings need quotes and escaping via `json.dumps`.
- **Allowing leading zeros or Unicode digits.** `-?[0-9]+` accepts `01`, and `\d` without `re.ASCII` matches non-ASCII digits.
- **Requiring a sign on exponents.** JSON allows unsigned exponents like `1e5`.
- **Allowing optional whitespace.** Inserting optional spaces after `:` or `,` matches non-compact JSON, and unbounded whitespace lets a model pad output indefinitely.
- **Using anchors or lookaround.** Features like `^`, `$`, or `(?=...)` work in `re.fullmatch` but are rejected by the DFA compiler.

## Complexity

The generated regex length is linear in the schema size with three multipliers. First, nested patterns are duplicated wherever they are pasted (twice per array level because `item` appears twice). Second, a run of $k$ leading optional properties in an object produces $O(k^2)$ member copies. Third, repetition bounds like `{0,100}` are tiny in the regex string but expand linearly into states during DFA compilation, so a string with `maxLength: 100` creates 100 copies of the string-item states. That third factor dominates in practice.

## In the wild

This compilation strategy comes from Outlines, introduced in "Efficient Guided Generation for Large Language Models" (arXiv 2307.09702) and implemented in `outlines-core` (`src/json_schema`). Like this solution, `outlines-core` recurses over the schema, fixes property order to declaration order, uses similar scalar patterns, and splits object serialization into two cases. It formulates the object split differently. When at least one property is required, properties before the last required one carry a trailing comma and those after it carry a leading comma. When no property is required, it emits one alternative for each choice of the last present property.

Three details differ from `outlines-core`:

- **Whitespace.** `outlines-core` inserts a configurable whitespace pattern between tokens (defaulting to `[ ]?`). This problem produces compact JSON, equivalent to setting that whitespace pattern to the empty string.
- **Numbers.** `outlines-core` uses `((-)?(0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?`, which requires an explicit sign on exponents. This problem follows RFC 8259, where the exponent sign is optional.
- **Strings.** `outlines-core` uses `[^"\\\x00-\x1F\x7F-\x9F]|\\["\\/bfnrt]`, which omits `\uXXXX` escapes and excludes U+007F to U+009F. This problem follows RFC 8259, which permits both.

Outlines also supports additional JSON Schema features beyond this subset that are not covered here.
