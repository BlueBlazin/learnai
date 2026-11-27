## The idea

Most of JSON is regular. Strings, numbers, the three literals, and the punctuation between them can each be recognized by a small finite automaton. What is not regular is arbitrary nesting. To decide whether `}` or `]` is valid, the recognizer must remember which container is innermost and, after closing it, which container was open before it. A finite automaton has a fixed number of states and cannot track unbounded depth. Pairing it with a stack solves this and produces a **pushdown automaton**.

The validator state therefore has two parts:

- A **mode** (the finite control) that tracks what the recognizer is currently inside or expects next.
- A **stack** of open containers. Only the top entry is inspected, determining whether a comma leads to an object key or an array value and which closing bracket is legal.

Prefix validation falls out automatically from this design. In every mode, `json_step` accepts only characters permitted by the grammar, and every reachable state has at least one valid completion (closing the current token, supplying a missing value, and closing any open containers). As a result, reaching a state without raising `Invalid` is equivalent to being a prefix of some valid JSON text, and `Invalid` is raised at the first impossible character.

Numbers are the one token type without a closing delimiter. After reading `12`, the number may be finished or it may continue with `3`, `.`, or `e`. The recognizer stays in a number mode until a character arrives that cannot continue the number. At that point, it checks that the number so far is complete (`1.` and `-` are not), transitions to the post-value mode, and processes that same character as whatever may follow a value. `json_complete` must also recognize complete number modes, because a top-level `12` is already a complete JSON text even though no delimiter has closed it.

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

Here the stack is stored as a Python string, so pushing and popping copy it in time proportional to the nesting depth while keeping every state an immutable tuple. A persistent linked stack of `(top, rest_of_stack)` pairs makes push and pop $O(1)$ and lets branching states share common prefixes. Grammar engines use that persistent representation because they test many candidate tokens from the same state.

## Common mistakes

- **No stack.** A bare depth counter (or a stack whose top is never matched against the closer) accepts mismatched brackets like `[1}` and `{"a":1]`. A bare counter also cannot tell whether a key or a value is expected after a comma.
- **Leading zeros.** Treating an integer as any sequence of digits accepts `01`. A standalone `0` and a nonzero digit followed by more digits require separate modes.
- **Complete before any value.** Checking only that the stack is empty and no token is open wrongly marks the empty string as complete. The initial `"value"` mode must be distinct from `"after"`, or JSON mode will allow an empty response.
- **Complete in the middle of a number.** `-`, `1.`, `2e`, and `2e+` are valid prefixes, not complete numbers. Only the four finished number modes (`"zero"`, `"int"`, `"frac"`, `"exp"`) should satisfy `json_complete`.
- **Trailing commas.** Transitioning back to `"key_or_end"` or `"value_or_end"` after a comma allows a closing bracket right after the comma, letting `[1,]` and `{"a":1,}` slip through.
- **Raw control characters in strings.** A model generating multi-line text will often emit a literal newline inside a string. It must be rejected in favor of the two-character escape `\n`.
- **Rejecting surrogates.** RFC 8259 accepts `\u` followed by any four hex digits. Section 8.2 notes that unpaired surrogate escapes are grammatical even though downstream software may handle them inconsistently, so stricter Unicode policies belong in a higher layer. Rejecting the surrogate range outright also breaks emojis encoded as escaped surrogate pairs.
- **Mutating the state.** Using a Python `list` with in-place `append` and `pop` causes states to alias the same list. The validator appears to work on a linear pass through one string, then corrupts state as soon as a caller branches to test multiple candidate tokens from the same state.

## Complexity

Let $d$ be the current container nesting depth. Each call to `json_step` does $O(1)$ work plus the cost of updating the stack, which is $O(d)$ with a string or tuple stack and $O(1)$ with a persistent linked stack. Memory usage is $O(d)$ for the stack. Testing a candidate token takes one step per character, and computing a full token mask repeats this across the vocabulary. Real constrained-decoding engines avoid most of that work by combining a pushdown recognizer with the precomputed token index from `dfa-token-index`, since tokens that stay inside a string or number never touch the stack.

## In the wild

- **RFC 8259** (2017) defines the grammar implemented here. Starting with RFC 7159, any JSON value can be a top-level document, whereas the original RFC 4627 required an object or array.
- **llama.cpp** constrains generation with GBNF grammars and ships both a JSON grammar and a JSON Schema to GBNF converter. Its engine maintains a set of grammar-position stacks and advances them one code point at a time, generalizing this single-stack recognizer to nondeterministic grammars.
- **XGrammar** compiles grammars to byte-level pushdown automata and stores stacks in a persistent tree shared across states, making speculative token checks cheap.
- **llguidance** splits recognition the same way as this solution, pairing a regex-based lexer for strings, numbers, and punctuation with an Earley parser for nested structure.
- **Outlines** translates non-recursive JSON Schemas into regular expressions and uses a precomputed DFA token index (`dfa-token-index`). That works when a fixed schema has bounded nesting depth, whereas free-form JSON or recursive schemas require a stack.
- OpenAI-style `response_format: {"type": "json_object"}` enforces the same syntactic validity checked here. If generation hits `max_tokens` early, the output is a valid prefix with `json_complete` still `False`, which is why truncated JSON-mode responses fail to parse.
