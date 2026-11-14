## The idea

Rows of a batch are independent inside the model: row `b` of the logits depends on
row `b` of the tokens and of the cache, and on nothing else. So batched greedy
decoding is the single-sequence loop of `cached-greedy-decode` run on `B` rows in
lockstep, and the only new thing is that the rows disagree about when to stop.

That disagreement is resolved with one bit of state per row, `finished[b]`, and two
rules that use it differently:

- **Per row:** a finished row collects nothing. Its slot in the batch is still fed
  (a filler token, `eos_id`) because the cache and the token tensor must stay
  rectangular, and the model still returns logits for it, but they are ignored.
- **For the batch:** the loop ends when *every* row is finished, and it checks this
  before the next call, so no call is made whose outputs nobody reads.

The invariant of the loop: *before each argmax, the cache covers, for every running
row, exactly its sequence so far; and `rows[b]` is final for every finished row.*

The cost follows. The prompts go in once ($B \cdot P$ tokens). The longest row needs
$g_{\max} - 1$ further calls, and each of them carries all $B$ rows:
$B \cdot (P + g_{\max} - 1)$ tokens in $g_{\max}$ calls.

## Reference solution (PyTorch)

```python
def batched_greedy(lm, prompts, max_new_tokens, eos_id):
    logits, cache = lm(prompts)
    rows = [[] for _ in prompts]
    finished = [False] * len(prompts)
    while True:
        tokens = torch.argmax(logits[:, -1, :], dim=-1).tolist()
        for b, token in enumerate(tokens):
            if finished[b]:
                continue
            rows[b].append(token)
            finished[b] = token == eos_id or len(rows[b]) == max_new_tokens
        if all(finished):
            return rows
        feed = [[eos_id] if finished[b] else [rows[b][-1]] for b in range(len(rows))]
        logits, cache = lm(feed, cache)
```

One `argmax` and one `.tolist()` per step move the `B` tokens to Python; the per-row
bookkeeping is a loop over `B` small integers, which is nothing next to a forward
pass. A row can only be finished at the `feed` line if it emitted EOS (rows that run
out of budget do so together, and then the function has returned), so `eos_id` is
never `None` there. Engines keep `finished` as a boolean tensor and write
`torch.where(finished, eos_id, tokens)`; the logic is the same.

## Common mistakes

- **`any` instead of `all`.** Ending the loop when the first row emits EOS cuts every
  other row short. With one row the two are the same, which is how this survives
  testing with batch size 1.
- **No sticky per-row flag.** If "finished" is recomputed from the current step's
  tokens, or rows keep appending until the loop ends, a row that emitted EOS at step 2
  returns `[16, 18, 33, 56, 31, ...]`: the model's continuation past EOS. In a real
  server this is the response that ends properly and then goes on with garbage.
- **Never stopping early.** A `for` loop over `max_new_tokens` steps returns the right
  tokens when the rows are trimmed afterwards, but calls the model long after the
  last row finished: 40 calls instead of 9 in one of the tests.
- **Checking "all finished" after the forward call.** One wasted batched call per
  generation; with short answers that is a noticeable share of the work.
- **Decoding the rows one at a time.** Right tokens, and the number of tokens
  processed is even lower, but the number of calls is $\sum_b g_b$ instead of
  $\max_b g_b$. On a GPU the calls are what cost time; this throws the batch away.
- **Feeding a finished row its own argmax.** The model then generates a real
  continuation for a row nobody is reading. The other rows are unaffected, which is
  why it goes unnoticed, but the convention matters as soon as anything inspects the
  batch: padding tokens are what attention masks, logging, and token accounting
  expect to see in a finished slot.

## Complexity

$g_{\max}$ forward calls and $B \cdot (P + g_{\max} - 1)$ tokens processed. Of
those, $\sum_b (g_{\max} - g_b)$ tokens are filler: work spent on rows that had
already finished. In the worked example that is $7 + 3 + 0 = 10$ of 36 tokens.

## In the wild

This is static batching. Hugging Face `generate` keeps an `unfinished_sequences`
vector, overwrites the next tokens of finished rows with `pad_token_id`
(`next_tokens * unfinished + pad * (1 - unfinished)`), and stops when the vector is
all zeros. Most offline batch-inference scripts do the same. The filler work is the price of the rectangle, and it grows with the spread of
the answer lengths. Serving engines (vLLM, TGI, llama.cpp's server slots) therefore
do not wait for the longest row: with *continuous batching* a finished row's slot is
given to the next waiting request at the following step, which needs a cache that
can drop and add rows between calls. Real prompts also differ in length, which adds
padding and attention masks on the prompt side; that is `padded-batch-invariance`
and, on MLX, `mlx-batched-generate`.
