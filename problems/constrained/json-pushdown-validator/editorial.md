## The idea

Most of JSON is regular. Strings, numbers, the three literals, and the punctuation
between them can each be recognized by a small finite automaton. What is not regular
is nesting: to know whether `}` or `]` is legal you must know which bracket is
innermost, and after closing it, which one was open before, to any depth. A finite
automaton has a fixed number of states and cannot count unbounded depth. Add a stack
and it can: that is a **pushdown automaton**.

So the state has two parts.

- A **mode**, the finite control: what the automaton is in the middle of, or what it
  expects next.
- A **stack** of open containers. Only its top is ever inspected: it decides whether
  a comma leads to a key or to a value, and which closer is legal.

The property that makes this a *prefix* validator comes for free. In every mode the
step function accepts exactly the characters that the grammar allows next, and from
every state that is reached there is a way to finish (close the token, supply the
missing value, close the containers). So "no `Invalid` so far" is the same as "a
prefix of some valid text", and the first impossible character is where `Invalid` is
raised.

Numbers are the one token without a terminator. `12` may be complete or may continue
with `3`, `.`, or `e`. The automaton stays in a number mode, and when a character
arrives that cannot continue the number, it first checks that the number was
complete (`1.` and `-` are not), then treats the character as whatever may follow a
value. `json_complete` has to know about these modes too: at the top level `12` is a
complete text although nothing has "closed" it.

## Reference solution (Python)

```python
class Invalid(ValueError):
    pass


WHITESPACE = " \t\n\r"
DIGITS = "0123456789"
HEX_DIGITS = "0123456789abcdefABCDEF"
SIMPLE_ESCAPES = '"\\/bfnrt'
LITERALS = {"t": "rue", "f": "alse", "n": "ull"}
NUMBER_DONE = ("zero", "int", "frac", "exp")     # the number so far is complete


def json_start():
    return ("value", "", None)                   # (mode, stack, rest)


def json_complete(state):
    mode, stack, _ = state
    return not stack and (mode == "after" or mode in NUMBER_DONE)


def json_step(state, ch):
    mode, stack, rest = state

    if mode == "string":                         # rest: is this string an object key?
        if ch == '"':
            return ("colon" if rest else "after", stack, None)
        if ch == "\\":
            return ("escape", stack, rest)
        if ch < " ":
            raise Invalid("control character in a string")
        return state
    if mode == "escape":
        if ch == "u":
            return ("hex", stack, (rest, 4))
        if ch in SIMPLE_ESCAPES:
            return ("string", stack, rest)
        raise Invalid("invalid escape")
    if mode == "hex":                            # rest: (is_key, digits left)
        is_key, left = rest
        if ch not in HEX_DIGITS:
            raise Invalid("not a hex digit")
        return ("string", stack, is_key) if left == 1 else ("hex", stack, (is_key, left - 1))

    if mode == "literal":                        # rest: the characters still to come
        if ch != rest[0]:
            raise Invalid("misspelled literal")
        return ("literal", stack, rest[1:]) if len(rest) > 1 else ("after", stack, None)

    if mode == "minus":
        if ch == "0":
            return ("zero", stack, None)
        if ch in DIGITS:
            return ("int", stack, None)
        raise Invalid("expected a digit")
    if mode == "int" and ch in DIGITS:           # not "zero": nothing follows a leading 0
        return state
    if mode in ("zero", "int") and ch == ".":
        return ("dot", stack, None)
    if mode in ("zero", "int", "frac") and ch in "eE":
        return ("exp_mark", stack, None)
    if mode in ("dot", "frac") and ch in DIGITS:
        return ("frac", stack, None)
    if mode == "exp_mark" and ch in "+-":
        return ("exp_sign", stack, None)
    if mode in ("exp_mark", "exp_sign", "exp") and ch in DIGITS:
        return ("exp", stack, None)
    if mode in ("dot", "exp_mark", "exp_sign"):
        raise Invalid("expected a digit")
    if mode in NUMBER_DONE:
        mode = "after"                           # the number ended before ch

    if ch in WHITESPACE:
        return (mode, stack, None)
    if mode == "after":
        if not stack:
            raise Invalid("text after the end of the document")
        in_object = stack[-1] == "{"
        if ch == ",":
            return ("key" if in_object else "value", stack, None)
        if ch == ("}" if in_object else "]"):
            return ("after", stack[:-1], None)
        raise Invalid("expected ',' or the closing bracket")
    if mode == "colon":
        if ch == ":":
            return ("value", stack, None)
        raise Invalid("expected ':'")
    if mode in ("key", "key_or_end"):
        if ch == '"':
            return ("string", stack, True)
        if ch == "}" and mode == "key_or_end":
            return ("after", stack[:-1], None)
        raise Invalid("expected an object key")

    if ch == "]" and mode == "value_or_end":     # mode is "value" or "value_or_end"
        return ("after", stack[:-1], None)
    if ch == '"':
        return ("string", stack, False)
    if ch == "{":
        return ("key_or_end", stack + "{", None)
    if ch == "[":
        return ("value_or_end", stack + "[", None)
    if ch == "-":
        return ("minus", stack, None)
    if ch == "0":
        return ("zero", stack, None)
    if ch in DIGITS:
        return ("int", stack, None)
    if ch in LITERALS:
        return ("literal", stack, LITERALS[ch])
    raise Invalid("expected a value")
```

The stack is a string, so pushing and popping copy it. That keeps every state a plain
immutable tuple at a cost proportional to the depth. A persistent stack, a linked list
of `(top, rest_of_stack)` pairs, makes both operations constant time and lets all the
states that share a prefix share memory; grammar engines use that representation
because they keep one stack per candidate token.

## Common mistakes

- **No stack.** A depth counter, or a stack whose top is never compared with the
  closing bracket, knows that something is open, not what. `[1}` and `{"a":1]` pass,
  and with a bare counter the automaton cannot even tell whether a key or a value is
  due after a comma.
- **Leading zeros.** Treating a number as "digits" accepts `01`. The grammar has two
  integer forms, `0` alone or a nonzero digit followed by digits, and they need two
  modes.
- **Complete before any value.** "Depth zero and not inside a token" is also true of
  the empty text. The start state must be distinct from the state after a value;
  otherwise a JSON mode lets the model answer with nothing.
- **Complete in the middle of a number.** `-`, `1.`, `2e`, and `2e+` are prefixes of
  numbers, not numbers. Only the modes in which the number could end count.
- **Trailing commas.** Reusing the "just opened" mode after a comma is tempting,
  because both expect an element, but that mode also accepts the closer.
- **Raw control characters in strings.** A model asked for multi-line text will happily
  emit a real line feed inside a string. It must be `\n`, two characters.
- **Rejecting surrogates.** The grammar is deliberately loose: `\u` plus four hex
  digits, whatever they are. RFC 8259 notes (section 8.2) that a text with an unpaired
  surrogate escape is grammatical and that software receiving it behaves
  unpredictably; a validator that follows the RFC accepts it, and a stricter policy
  belongs in a layer above. Rejecting the whole range also breaks every emoji written
  as an escaped surrogate pair.
- **Mutating the state.** With a list as the stack and `append` / `pop` in place, all
  states share one list. The validator works when driven straight through a text and
  fails as soon as a caller tries two continuations from one state, which is the only
  thing a token mask ever does.

## Complexity

`json_step` is $O(1)$ work plus the cost of the stack operation: $O(d)$ at depth $d$
with a string or tuple stack, $O(1)$ with a persistent linked stack. Memory is the
stack, $O(d)$. Validating a token costs one step per character, and a mask costs that
for every token in the vocabulary, which is why real engines combine a recognizer like
this one with the precomputation of `dfa-token-index`: tokens that stay inside one
string or number never touch the stack and can be resolved ahead of time.

## In the wild

- **RFC 8259** (2017) is the grammar used here. Since RFC 7159 a JSON text may be any
  value; the original RFC 4627 required an object or an array at the top level.
- **llama.cpp** constrains output with GBNF grammars and ships a JSON grammar and a
  converter from JSON Schema to GBNF. Its grammar engine keeps a set of stacks of
  grammar positions and advances them one code point at a time, the nondeterministic
  version of the single stack here.
- **XGrammar** compiles the grammar to a byte-level pushdown automaton and stores the
  stacks persistently (as a tree shared between states), precisely so that trying a
  token and backing out is cheap.
- **llguidance** splits the work the way this editorial does: a lexer built from
  regular expressions for strings, numbers, and punctuation, and an Earley parser for
  the nesting.
- **Outlines** takes the other road for JSON Schema: it translates the schema to a
  regular expression and uses the index of `dfa-token-index`. That works because a
  fixed schema without recursion has bounded depth; free-form JSON, or a recursive
  schema, does not, and needs the stack.
- OpenAI-style `response_format: {"type": "json_object"}` promises what this
  recognizer checks: the output is syntactically valid JSON. If generation is cut off
  by `max_tokens`, the text is a valid *prefix* but `json_complete` is false, which
  is why truncated JSON-mode responses fail to parse.
