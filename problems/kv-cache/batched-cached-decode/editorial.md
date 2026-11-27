## The idea

Rows of a batch are independent inside the model, so row `b` of the logits depends only on row `b` of the input tokens and the cache. Batched greedy decoding is therefore the single-sequence loop from `cached-greedy-decode` run across `B` rows in lockstep, except that rows can finish at different steps.

You resolve that with one boolean flag per row, `finished[b]`, and two rules:

- **Per row:** A finished row collects no more tokens. Its batch slot still receives a filler token (`eos_id`) so the cache and token tensor stay rectangular, and its output logits are ignored.
- **For the batch:** The loop terminates as soon as *every* row is finished. Checking this before the next model call avoids running a forward pass whose outputs nobody reads.

This maintains a clear loop invariant: *before each argmax, the cache holds the exact sequence generated so far for every active row, and `rows[b]` is final for every finished row.*

The cost follows directly. All prompts enter in the first call ($B \cdot P$ tokens). The longest row then needs $g_{\max} - 1$ additional calls, each carrying all $B$ rows, for a total of $g_{\max}$ calls and $B \cdot (P + g_{\max} - 1)$ tokens processed.

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

Each step runs one `argmax` and one `.tolist()` to move the `B` token IDs to Python. The per-row bookkeeping loops over `B` small integers, which is negligible compared to a forward pass. Note that at the `feed` line, a row can only be finished if it emitted `eos_id` (if rows hit `max_new_tokens`, they all do so together and `all(finished)` returns first), so `eos_id` is never `None` there. Production engines keep `finished` as a boolean tensor and compute `torch.where(finished, eos_id, tokens)`, which implements the same logic.

## Common mistakes

- **Using `any` instead of `all`.** Ending the loop when the first row emits EOS truncates every other row. With batch size 1 the two behave identically, so this bug easily survives single-row tests.
- **No sticky per-row flag.** Recomputing "finished" from only the current step's tokens, or appending until the loop ends, causes a row that emitted EOS at step 2 to return `[16, 18, 33, 56, 31, ...]`. In a real server, the response finishes cleanly and then appends garbage.
- **Never stopping early.** Running a `for` loop for all `max_new_tokens` steps and trimming afterwards returns the right tokens, but keeps calling the model long after every row has finished (40 calls instead of 9 in one of the tests).
- **Checking `all(finished)` after the forward call.** This wastes one full batched call on every generation, which is a noticeable overhead for short responses.
- **Decoding rows one at a time.** This produces the right tokens and even processes fewer total tokens, but makes $\sum_b g_b$ forward calls instead of $\max_b g_b$. On a GPU, forward calls dominate latency, so this throws away the benefit of batching.
- **Feeding a finished row its own argmax.** The model then generates a real continuation for a slot nobody reads. Other rows are unaffected, so the mistake is easy to miss, but downstream attention masks, logging, and token accounting expect filler tokens in finished slots.

## Complexity

The loop makes $g_{\max}$ forward calls and processes $B \cdot (P + g_{\max} - 1)$ tokens. Of those tokens, $\sum_b (g_{\max} - g_b)$ are filler spent on rows that have already finished. In the worked example, that is $7 + 3 + 0 = 10$ of the 36 tokens.

## In the wild

This pattern is static batching. Hugging Face `generate` tracks an `unfinished_sequences` vector, overwrites the next tokens of finished rows with `pad_token_id` (`next_tokens * unfinished + pad * (1 - unfinished)`), and stops when the vector is all zeros. Most offline batch-inference scripts do the same.

Filler work is the cost of keeping a rectangular batch, and it grows with the spread in response lengths. Serving engines such as vLLM, TGI, and `llama.cpp`'s server slots avoid waiting for the longest row by using *continuous batching*, which reassigns a finished row's slot to a waiting request on the very next step. That requires a cache that can drop and add rows between calls. Real prompts also vary in length, which adds prompt-side padding and attention masks—covered in `padded-batch-invariance` and, on MLX, `mlx-batched-generate`.
