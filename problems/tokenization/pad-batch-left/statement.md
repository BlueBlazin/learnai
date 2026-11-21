Prompts in a batch have different lengths, and a model needs a rectangle. Implement the
function that pads a batch of token sequences and reports which positions are real. The
detail that matters for generation is the side: a decoder-only model continues each row
from its **last** column, so the padding has to go on the left.

![This problem is the Pad batch step, making token sequences of different lengths into a rectangle.](assets/map.svg)

## Specification

```python
def pad_batch(
    seqs: list[list[int]], pad_id: int, side: str = "left"
) -> tuple[list[list[int]], list[list[int]], list[int]]: ...
```

Let $B$ be the number of sequences, $n_i$ the length of sequence $i$, and
$T = \max_i n_i$. The function returns the tuple `(ids, attention_mask, lengths)`:

- **`ids`**: $B$ rows of exactly $T$ token ids. Row $i$ holds the $n_i$ tokens of
  `seqs[i]`, in order and contiguous, and $T - n_i$ copies of `pad_id`.
  With `side="left"` the padding comes first and the tokens occupy the last $n_i$
  positions; with `side="right"` the tokens come first and the padding last.
- **`attention_mask`**: $B$ rows of $T$ ints, `1` at the positions of `ids` that hold a
  token of the sequence and `0` at the positions that hold padding.
- **`lengths`**: the list $[n_0, \dots, n_{B-1}]$.

Rules:

- `side` defaults to `"left"`.
- The mask describes *positions*, not values. `pad_id` may also occur as a real token
  inside a sequence (many models pad with their end-of-sequence token); such a token
  has mask `1`.
- A sequence with no tokens is an error: raise `ValueError` if any sequence is empty.
  Its row would be all padding with an all-zero mask, which gives the model nothing to
  attend to and nothing to continue.
- Raise `ValueError` if `seqs` is empty, and if `side` is any string other than
  `"left"` or `"right"` (the comparison is exact: `"Left"` is an error). The errors do
  not depend on whether any row actually needs padding.
- Everything returned is new and built from plain Python values: `ids` and
  `attention_mask` are lists of lists of `int`, `lengths` is a list of `int`. The
  function does not modify `seqs` or its sequences, and no row of `ids` is the same
  list object as an input sequence, also when that row needs no padding.

## Example

```python
ids, attention_mask, lengths = pad_batch([[5, 6, 7], [8], [9, 10]], pad_id=0)
ids             # [[5, 6, 7], [0, 0, 8], [0, 9, 10]]
attention_mask  # [[1, 1, 1], [0, 0, 1], [0, 1, 1]]
lengths         # [3, 1, 2]
```

The last column, `7, 8, 10`, is the last real token of every prompt: the logits at
position $T - 1$ are the next-token logits for all three rows. With right padding that
column would be mostly padding:

```python
ids, attention_mask, lengths = pad_batch([[5, 6, 7], [8], [9, 10]], pad_id=0, side="right")
ids             # [[5, 6, 7], [8, 0, 0], [9, 10, 0]]
attention_mask  # [[1, 1, 1], [1, 0, 0], [1, 1, 0]]
lengths         # [3, 1, 2]
```

When the pad token is also a real token, here `2`, only the mask tells them apart:

```python
ids, attention_mask, lengths = pad_batch([[7, 2, 9, 2], [2], [4, 2]], pad_id=2)
ids             # [[7, 2, 9, 2], [2, 2, 2, 2], [2, 2, 4, 2]]
attention_mask  # [[1, 1, 1, 1], [0, 0, 0, 1], [0, 0, 1, 1]]
lengths         # [4, 1, 2]
```

Errors:

```python
pad_batch([[1, 2], [], [3]], 0)          # ValueError: an empty sequence
pad_batch([], 0)                         # ValueError: an empty batch
pad_batch([[1], [2, 3]], 0, "center")    # ValueError: unknown side
```

## Constraints

- At most 64 sequences of at most 4096 tokens each. Token ids and `pad_id` are
  non-negative Python ints; `side` is a string.
- All three results are compared exactly, and their types are checked: lists of plain
  `int`, not booleans and not arrays.
- There is no speed requirement beyond the time limit.

## Why this matters

Batched generation appends each new token after the last column and reads the next
token's logits from the last position. Padding on the right puts pad tokens between a
short prompt and its continuation: the model is asked to continue a run of padding it
never saw in training, and the short rows of the batch produce garbage while the
longest row looks fine, which is why the bug survives single-prompt testing. Most
tokenizer libraries pad on the right by default, because that is what training and
encoder models want, so every generation server sets the side explicitly. The mask
matters just as much: it is what keeps the pad positions out of attention, and deriving
it from `ids != pad_id` silently hides real end-of-sequence tokens in models whose pad
token is their EOS token.
