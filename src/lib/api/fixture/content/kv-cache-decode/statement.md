Generating $n$ tokens after a prompt of length $L$ without a cache runs the model on $L + 1, L + 2, \dots, L + n$ tokens: the work grows quadratically. A key-value cache stores each layer's attention keys and values, so every step after the first processes exactly **one** new token.

You are given an instrumented model from the kit. Write the greedy decode loop that uses its cache correctly.

## The kit model

```python
class CountingModel:
    def forward(self, tokens: Tensor, cache: Cache | None) -> tuple[Tensor, Cache]:
        """
        tokens: (B, T) ids for the positions NOT already in `cache`.
        Returns logits (B, T, V) for those positions and the extended cache.
        """
```

Every call adds $B \cdot T$ to the model's token counter. The judge reads the counter after your function returns.

## Interface

```python
def greedy_decode(model: CountingModel, prompt: Tensor, max_new_tokens: int, eos_id: int) -> Tensor:
    """
    prompt: (1, L) token ids.
    Returns (1, L + n) ids: the prompt followed by n generated tokens,
    stopping after `eos_id` is produced or at `max_new_tokens`.
    """
```

## Cost model

| Strategy | Tokens processed | $L = 32$, $n = 5$ |
|---|---|---:|
| No cache | $\sum_{i=0}^{n-1} (L + i) = nL + \tfrac{n(n-1)}{2}$ | 170 |
| Cache, re-feeding the last prompt token | $L + n$ | 37 |
| Cache, optimal | $L + n - 1$ | 36 |

The limit is the optimal count, $L + n - 1$: the prompt once, then one token per step, and the final generated token is never fed back because nothing follows it.

### Example 1

With $L = 4$, `max_new_tokens = 3`, and no end-of-sequence token, the calls are

```python
logits, cache = model.forward(prompt, None)            # 4 tokens
logits, cache = model.forward(next_token, cache)       # 1 token
logits, cache = model.forward(next_token, cache)       # 1 token
# third token is taken from the last logits; no further call
```

for a total of $6 = 4 + 3 - 1$.

> [!NOTE]
> The output must also be **correct**: the numeric cases compare your ids with an uncached reference. A cache used wrongly is fast and wrong.

## Constraints

- $1 \le L \le 512$, $1 \le$ `max_new_tokens` $\le 128$, batch size is always $1$.
- Greedy means arg max of the last position's logits; ties go to the lower id.
