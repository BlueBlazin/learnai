Separate the problem into two questions: *where* special strings occur in `text`, and *what* to do with each match. Which of those two depends on `allowed`? Check the example where only `"<|sep|>"` is allowed but the text contains `"<|sep|>\n"`.

---

Track two indices: `pos` for the current scan position and `run_start` for the start of the current ordinary run. At each `pos`, test whether any key of `specials` matches via `text.startswith(key, pos)`, checking longer keys before shorter ones. If none matches, increment `pos`. If a key matches, raise `ValueError` unless it is in `allowed`, encode `text[run_start:pos]` if non-empty, append the special's id, advance `pos` past the special, and reset `run_start = pos`. Remember to encode any remaining tail after the loop.

---

Once the keys are sorted by descending length, finding the longest match at `pos` is one expression:

```python
by_length = sorted(specials, key=len, reverse=True)
special = next((s for s in by_length if text.startswith(s, pos)), None)
```

Handle `allowed == "none_as_text"` up front by returning `encode_ordinary(text)`. After the loop, flush the remaining ordinary text with `if run_start < len(text): ids.extend(encode_ordinary(text[run_start:]))`.
