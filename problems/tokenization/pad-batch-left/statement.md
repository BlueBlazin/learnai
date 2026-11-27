Implement `pad_batch` to pad a batch of variable-length token sequences into a rectangular 2D list and return an attention mask marking the real tokens. Because a decoder-only model continues each prompt from its **last** column during batched generation, padding goes on the left by default.

![This problem is the Pad batch step, making token sequences of different lengths into a rectangle.](assets/map.svg)

## Specification

```python
def pad_batch(
    seqs: list[list[int]], pad_id: int, side: str = "left"
) -> tuple[list[list[int]], list[list[int]], list[int]]: ...
```

Let $B$ be the number of sequences in `seqs`, $n_i$ be the length of `seqs[i]`, and $T = \max_i n_i$ be the length of the longest sequence. Return the tuple `(ids, attention_mask, lengths)`:

- **`ids`**: $B$ rows of exactly $T$ token ids. Row $i$ contains the $n_i$ tokens of `seqs[i]` in order and contiguous, along with $T - n_i$ copies of `pad_id`. When `side == "left"`, padding comes first and the real tokens occupy the last $n_i$ positions. When `side == "right"`, the real tokens come first and padding comes last.
- **`attention_mask`**: $B$ rows of $T$ ints, with `1` at positions that hold a real token from `seqs[i]` and `0` at positions that hold padding.
- **`lengths`**: the list $[n_0, \dots, n_{B-1}]$.

Rules:

- `side` defaults to `"left"`.
- The mask tracks *positions*, not token values. Many models pad with their end-of-sequence token, so `pad_id` can appear as a real token inside `seqs[i]`. Every real token must have mask `1`.
- Raise `ValueError` if any sequence in `seqs` is empty. An empty sequence would produce an all-padding row with an all-zero mask, leaving the model nothing to attend to and no token to continue from.
- Raise `ValueError` if `seqs` is empty or if `side` is any string other than `"left"` or `"right"`. The comparison is exact, so `"Left"` raises `ValueError`. Validate all inputs even if no row needs padding.
- Return new plain Python objects. `ids` and `attention_mask` must be lists of lists of `int`, and `lengths` must be a list of `int`. Do not modify `seqs` or its inner lists, and do not reuse an input list as a row in `ids` even when that row needs no padding.

## Example

```python
ids, attention_mask, lengths = pad_batch([[5, 6, 7], [8], [9, 10]], pad_id=0)
ids             # [[5, 6, 7], [0, 0, 8], [0, 9, 10]]
attention_mask  # [[1, 1, 1], [0, 0, 1], [0, 1, 1]]
lengths         # [3, 1, 2]
```

With left padding, the last column (`7, 8, 10`) holds the final real token of every prompt. That means the logits at position $T - 1$ give the next-token predictions for all three rows. With right padding, that last column is mostly padding instead:

```python
ids, attention_mask, lengths = pad_batch([[5, 6, 7], [8], [9, 10]], pad_id=0, side="right")
ids             # [[5, 6, 7], [8, 0, 0], [9, 10, 0]]
attention_mask  # [[1, 1, 1], [1, 0, 0], [1, 1, 0]]
lengths         # [3, 1, 2]
```

When `pad_id` also appears as a real token (here `2`), only `attention_mask` distinguishes real tokens from padding:

```python
ids, attention_mask, lengths = pad_batch([[7, 2, 9, 2], [2], [4, 2]], pad_id=2)
ids             # [[7, 2, 9, 2], [2, 2, 2, 2], [2, 2, 4, 2]]
attention_mask  # [[1, 1, 1, 1], [0, 0, 0, 1], [0, 0, 1, 1]]
lengths         # [4, 1, 2]
```

Invalid inputs raise `ValueError`:

```python
pad_batch([[1, 2], [], [3]], 0)          # ValueError: an empty sequence
pad_batch([], 0)                         # ValueError: an empty batch
pad_batch([[1], [2, 3]], 0, "center")    # ValueError: unknown side
```

## Constraints

- At most 64 sequences, each with at most 4096 tokens.
- Token ids and `pad_id` are non-negative Python ints, and `side` is a string.
- All three returned values are compared exactly. Their types are also checked and must be lists of plain `int`, not booleans or arrays.
- There is no speed requirement beyond the time limit.

## Why this matters

Batched generation reads the next-token logits from the last column and appends each newly sampled token right after it. If you pad on the right, short prompts end with a block of pad tokens between the prompt and its continuation. The model then tries to continue a run of padding it never saw during training, so short rows produce garbage while the longest row looks fine. Because single-prompt tests have no padding, this bug easily slips past unit tests.

Most tokenizer libraries default to right padding because training and encoder models expect it, so generation servers must set the padding side explicitly. Building the attention mask from sequence lengths is equally important. Computing the mask as `ids != pad_id` silently masks out real end-of-sequence tokens in models that reuse their EOS token for padding.
