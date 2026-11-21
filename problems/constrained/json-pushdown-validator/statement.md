Write a JSON recognizer that reads one character at a time and can be stopped and
asked, at any point, "can this still become valid JSON?" and "is it valid JSON
already?". Brackets nest to any depth, so no finite automaton can do it: the state
needs a stack. This is the machine a JSON mode runs for every candidate token.

## Specification

```python
class Invalid(ValueError): ...

def json_start(): ...
def json_step(state, ch): ...
def json_complete(state) -> bool: ...
```

- **`json_start()`** returns the state before any character has been read.
- **`json_step(state, ch)`** reads the character `ch` (a `str` of length 1). If the
  text read so far, including `ch`, is a **prefix of at least one valid JSON text**,
  it returns the new state. Otherwise it raises `Invalid`. So `Invalid` is raised at
  the first character after which no continuation could make the text valid, and
  never earlier.
- **`json_complete(state)`** returns `True` if the text read so far is, as it stands,
  a complete valid JSON text, and `False` otherwise.

A state may be any value. It is opaque to the tests, with one requirement:
**`json_step` must not modify the state it is given.** The caller may keep a state,
step it with one character, and later step the same state with another one. (A
token mask is computed exactly like that: every candidate token is tried from the
same state.) Build states from immutable parts.

`Invalid` is the exception class defined in the starter. Keep it, and raise it (or a
subclass) for every rejection.

### The grammar (RFC 8259)

A JSON text is one value with optional whitespace around it.

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

Points that are easy to get wrong:

- **Any value may be the whole text**, not only an object or an array: `7`, `"x"`,
  and `null` are complete JSON texts. The empty text and whitespace alone are not.
- **Numbers.** A leading zero stands alone: `0`, `-0`, `0.5`, and `0e1` are numbers,
  `01` and `-007` are not. A fraction and an exponent need at least one digit: `1.`
  and `1e` are not numbers (but they are prefixes of numbers). There is no leading
  `+`, no `.5`, no `NaN`, and no `Infinity`.
- **A number has no closing character.** It ends when a character arrives that
  cannot continue it, and that character is then judged as what follows a value.
  Consequently, after `12` at the top level `json_complete` is `True` *and* a further
  digit, `.`, or `e` is accepted. After `12.` it is `False`.
- **Commas separate.** `[1,]`, `{"a":1,}`, `[,1]`, and `[1,,2]` are invalid.
- **Brackets match.** `}` closes only an object and `]` only an array, the innermost
  open one. After a comma inside an object a key string must follow; inside an array,
  a value.
- **Object keys are strings.** Duplicate keys are allowed (the grammar does not look
  at them).
- **Strings.** The control characters U+0000 to U+001F must be escaped; a raw line
  feed or tab inside a string is invalid. Every other character is allowed as it is,
  including U+007F and all non-ASCII characters. After `\u`, *any* four hex digits
  are accepted. Surrogates get no special treatment: `"\ud800"` (a lone surrogate) is
  valid JSON by the grammar, and so is `"😀"`.
- **Literals** are exactly `true`, `false`, `null`, in lowercase.
- **Whitespace** is only those four characters. A byte order mark, a non-breaking
  space, a vertical tab, or a comment is invalid wherever it appears outside a string.
- **After a complete top-level value** only whitespace may follow.

Write the recognizer yourself. Do not use the `json` module or any other parser; the
harness rejects `json.loads`, `json.load`, and `json.JSONDecoder`.

## Example

Reading `{"a": [1, 2.5e3, true], "b": null}` character by character never raises.
`json_complete` is `False` after every character except the last one:

| text read so far | open containers | `json_complete` |
|---|---|---|
| (nothing) | | `False` |
| `{"a": [1` | `{` `[` | `False` |
| `{"a": [1, 2.5e3` | `{` `[` | `False` |
| `{"a": [1, 2.5e3, true]` | `{` | `False` |
| `{"a": [1, 2.5e3, true], "b": null` | `{` | `False` |
| `{"a": [1, 2.5e3, true], "b": null}` | | `True` |

From the final state a space is accepted (and the text stays complete), and any other
character raises `Invalid`.

![Five rows of the table. The stack holds the open containers, the innermost one last.](assets/stack-steps.svg)

```python
state = json_start()
for ch in '{"a": [1, 2.5e3, true':
    state = json_step(state, ch)

json_step(state, "]")      # fine: closes the array
json_step(state, ",")      # fine, from the same state: another element may follow
json_step(state, "}")      # raises Invalid: the innermost open container is an array
```

Rejections happen at the first character that rules out every continuation:

```python
'{"a": [1, 02]}'           # Invalid at the "2": after a leading 0 no digit may follow
'[1,]'                     # Invalid at the "]"
'[1.]'                     # Invalid at the "]": "1." needs a digit ("[1." was a fine prefix)
'"a<line feed>b"'          # Invalid at the raw line feed
'tru'                      # never Invalid; json_complete is False
```

## Constraints

- Texts have at most 10,000 characters and nest at most 1,000 containers deep. `ch`
  can be any Unicode character.
- Verdicts are exact: the tests check which characters are accepted at every prefix of
  many texts, the index at which `Invalid` is raised for invalid ones, and
  `json_complete` at every prefix. `json_complete` is also compared with a real JSON
  parser on whole texts.
- `json_step` should take time independent of the length of the text read so far
  (pushing to and popping from the stack aside).

## Why this matters

"JSON mode" means the server guarantees that the output parses. It does so by asking,
for each candidate token, whether the text would still be a prefix of valid JSON,
which is `json_step` over the token's characters, and by allowing the model to stop
only where `json_complete` is true. Regular expressions and DFAs, the tools of the
previous two problems, cannot express this, because matching brackets to arbitrary
depth needs unbounded memory. The stack is the entire difference between a regex
constraint and a grammar constraint, and this recognizer is the smallest real example
of the pushdown automata that llama.cpp grammars, XGrammar, and llguidance run. Each
"easy to get wrong" point above is a way to hand a client a response that its JSON
parser refuses.
