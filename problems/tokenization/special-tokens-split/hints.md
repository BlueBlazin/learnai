There are two separate questions: *where* are the special strings in the text, and
*what happens* to each one you find. Which of the two depends on `allowed`? Look at the
example where only `"<|sep|>"` is allowed but the text contains `"<|sep|>\n"`.

---

Keep two indices: `pos`, where you are looking now, and `run_start`, where the current
run of ordinary text began. At each `pos`, ask whether any key of `specials` starts
there (`text.startswith(key, pos)`), trying longer keys before shorter ones. If none
does, advance `pos` by one. If one does: it must be in `allowed`, or you raise; then
encode `text[run_start:pos]` if it is not empty, append the special's id, jump `pos`
past the special, and start a new run there. After the loop, one last run may be left.

---

The search for the special at one position is a single line once the keys are sorted
by length:

```python
by_length = sorted(specials, key=len, reverse=True)
special = next((s for s in by_length if text.startswith(s, pos)), None)
```

Handle `allowed == "none_as_text"` before anything else by returning
`encode_ordinary(text)`. And do not forget the tail: `if run_start < len(text):
ids.extend(encode_ordinary(text[run_start:]))`.
