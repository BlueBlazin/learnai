Implement an incremental JSON validator that reads one character at a time. At any point, the caller can ask whether the text read so far can still become valid JSON and whether it is already a complete JSON text. Because JSON brackets can nest to arbitrary depth, a finite automaton is not enough, and the validator state must maintain a stack.

## Specification

```python
class Invalid(ValueError): ...

def json_start(): ...
def json_step(state, ch): ...
def json_complete(state) -> bool: ...
```

- **`json_start()`** returns the initial state before any character has been read.
- **`json_step(state, ch)`** consumes a single character `ch` (a `str` of length 1). If the text read so far, including `ch`, is a **prefix of at least one valid JSON text**, it returns the next state. Otherwise it raises `Invalid`. Rejection must happen at the exact first character after which no continuation can make the text valid, and never earlier.
- **`json_complete(state)`** returns `True` if the text read so far is already a complete valid JSON text, and `False` otherwise.

The returned `state` can be any opaque value, except that **`json_step` must not mutate the state it receives.** A caller may save a state, step it with one character, and later step that same state with a different character. Constrained decoding computes a token mask in exactly this way by testing every candidate token from the same state. Build your states from immutable values.

Keep the `Invalid` exception class defined in the starter, and raise `Invalid` (or a subclass) for every rejection.

### The grammar (RFC 8259)

A valid JSON text is a single value surrounded by optional whitespace.

```
text    = ws value ws
ws      = zero or more of: space, tab (\t), line feed (\n), carriage return (\r)
value   = object | array | string | number | "true" | "false" | "null"
object  = "{" ws "}"   |   "{" member ("," member)* "}"
member  = ws string ws ":" ws value ws
array   = "[" ws "]"   |   "[" ws value ws ("," ws value ws)* "]"
number  = "-"? int frac? exp?
int     = "0" | digit1-9 digit*
frac    = "." digit+
exp     = ("e" | "E") ("+" | "-")? digit+
string  = '"' char* '"'
char    = any character except '"', '\', and U+0000 to U+001F
        | '\' followed by one of  " \ / b f n r t
        | '\u' followed by exactly four hex digits (0-9, a-f, A-F)
```

Watch out for these rules:

- **Top-level values.** Any JSON value may be the entire document, not just an object or an array. For example, `7`, `"x"`, and `null` are complete JSON texts, whereas an empty string or whitespace alone is not.
- **Numbers.** A leading zero must stand alone. `0`, `-0`, `0.5`, and `0e1` are valid numbers, while `01` and `-007` are invalid. Fractional parts and exponents require at least one digit, so `1.` and `1e` are valid prefixes but not complete numbers. Leading `+`, leading `.` (such as `.5`), `NaN`, and `Infinity` are invalid.
- **Number termination.** A number has no closing delimiter. It ends when a character arrives that cannot continue the number, and that character is then validated as whatever may follow a value. After reading `12` at the top level, `json_complete` is `True` and a subsequent digit, `.`, or `e` is still accepted. After reading `12.`, `json_complete` is `False`.
- **Commas.** Commas separate elements and members. Trailing, leading, or repeated commas such as `[1,]`, `{"a":1,}`, `[,1]`, and `[1,,2]` are invalid.
- **Matching brackets.** `}` closes only an object and `]` closes only an array, matching the innermost open container. After a comma inside an object, a key string must follow. Inside an array, a value must follow.
- **Object keys.** Keys must be strings. Duplicate keys are allowed because the grammar does not inspect key values.
- **Strings.** Control characters from `U+0000` through `U+001F` must be escaped, so a raw line feed or tab inside a string is invalid. Every other character is allowed unescaped, including `U+007F` and all non-ASCII characters. After `\u`, any four hex digits are accepted. Surrogates receive no special handling, so both `"\ud800"` (a lone surrogate) and `"😀"` are valid JSON by the grammar.
- **Literals.** The only literals are `true`, `false`, and `null`, all in lowercase.
- **Whitespace.** Only space, `\t`, `\n`, and `\r` are valid whitespace. A byte order mark, non-breaking space, vertical tab, or comment is invalid anywhere outside a string.
- **Trailing content.** After a complete top-level value, only whitespace may follow.

Write the validator yourself without using the `json` module or any other JSON parser. The harness rejects `json.loads`, `json.load`, and `json.JSONDecoder`.

## Example

Reading `{"a": [1, 2.5e3, true], "b": null}` one character at a time never raises `Invalid`. `json_complete` stays `False` after every character until the closing `}`:

| text read so far | open containers | `json_complete` |
|---|---|---|
| (nothing) | | `False` |
| `{"a": [1` | `{` `[` | `False` |
| `{"a": [1, 2.5e3` | `{` `[` | `False` |
| `{"a": [1, 2.5e3, true]` | `{` | `False` |
| `{"a": [1, 2.5e3, true], "b": null` | `{` | `False` |
| `{"a": [1, 2.5e3, true], "b": null}` | | `True` |

From the final state, whitespace is still accepted (and the text remains complete), while any other character raises `Invalid`.

![Five rows of the table. The stack holds the open containers, the innermost one last.](assets/stack-steps.svg)

```python
state = json_start()
for ch in '{"a": [1, 2.5e3, true':
    state = json_step(state, ch)

json_step(state, "]")      # valid: closes the array
json_step(state, ",")      # valid from the same state: another element may follow
json_step(state, "}")      # raises Invalid: the innermost open container is an array
```

Invalid inputs fail at the first character that rules out every valid continuation:

```python
'{"a": [1, 02]}'           # Invalid at the "2": no digit may follow a leading 0
'[1,]'                     # Invalid at the "]"
'[1.]'                     # Invalid at the "]": "1." needs a digit ("[1." is a valid prefix)
'"a<line feed>b"'          # Invalid at the raw line feed
'tru'                      # never Invalid; json_complete is False
```

## Constraints

- Texts have at most 10,000 characters and nest at most 1,000 containers deep. `ch` can be any Unicode character.
- Verdicts must be exact. Tests check which characters are accepted at every prefix of many texts, the exact index where `Invalid` is raised for invalid texts, and `json_complete` at every prefix. `json_complete` is also verified against a real JSON parser on whole texts.
- `json_step` should run in time independent of the length of the text read so far, aside from pushing to and popping from the stack.

## Why this matters

In LLM serving, "JSON mode" guarantees that the generated output parses as valid JSON. The server enforces this by testing each candidate token with `json_step` over the token's characters and allowing end-of-sequence only when `json_complete` is `True`. Regular expressions and finite automata cannot express JSON because matching nested brackets to arbitrary depth requires unbounded memory. Adding a stack turns a finite automaton into a pushdown automaton, which is the core mechanism behind grammar-constrained decoding in llama.cpp, XGrammar, and llguidance. Missing any edge case in the grammar above lets the model emit output that a downstream JSON parser will reject.
