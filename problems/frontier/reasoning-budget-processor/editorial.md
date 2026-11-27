## The idea

A logits processor maps what has been generated so far and the model's next-token logits to the subset of tokens allowed next. This processor needs only two pieces of state, a phase and a single counter `n`. The four phases form a pipeline with two shortcuts to *answer*:

```
start ──start_id──> thinking ──budget used up──> forcing ──end sequence done──> answer
  │                    │                                                          ▲
  └── any other token ─┼──────────────── end_id (the model closes) ───────────────┘
```

Each phase has one masking rule. Both *start* and *answer* apply `answer_mask` when present, with *start* also allowing `start_id` so constrained requests can still think. *thinking* leaves all logits untouched, and *forcing* keeps only `forced[n]`.

Two details determine correctness.

**What counts.** The budget counts only the thinking tokens that the model chose freely inside the span. It excludes `start_id`, the tokens of `forced`, and any `end_id` emitted by the model itself. Under this definition, the model emits at most `budget` thinking tokens in every case, including `budget = 0`, where `forced` follows `start_id` immediately.

**When to check the budget.** The call that receives the final allowed thinking token must return the first forced logits row right away. That means checking `phase == "thinking" and n == budget` after advancing the state, inside the same call. Checking before the transition, or using `>` instead of `==`, triggers one token too late.

Forcing works by masking rather than overwriting values. Setting every other entry to `-inf` while keeping the forced token's raw logit unchanged guarantees that any downstream sampler (greedy, temperature, or top-p) selects that token without special cases and preserves its true log-probability.

## Reference solution (PyTorch)

```python
def budget_step(state, token, logits, cfg):
    forced = cfg["forced"]
    mask = cfg["answer_mask"]

    if state is None:
        phase, n = "start", 0
    else:
        phase, n = state
        if phase == "start":
            phase = "thinking" if token == cfg["start_id"] else "answer"
        elif phase == "thinking":
            if token == cfg["end_id"]:
                phase = "answer"
            else:
                n += 1
        elif phase == "forcing":
            n += 1
            if n == len(forced):
                phase = "answer"
        if phase == "thinking" and n == cfg["budget"]:
            phase, n = "forcing", 0

    ids = torch.arange(logits.shape[-1])
    if phase == "forcing":
        keep = ids == forced[n]
    elif phase == "answer" and mask is not None:
        keep = mask
    elif phase == "start" and mask is not None:
        keep = mask | (ids == cfg["start_id"])
    else:
        return (phase, n), logits
    return (phase, n), torch.where(keep, logits, float("-inf"))
```

Because the state is an immutable `(phase, n)` tuple, `budget_step` is pure. The same code works in JAX and MLX by replacing `torch.arange` and `torch.where` with `jnp` or `mx`. Both the state update and the masking take only a few elementwise operations and never invoke the model.

## Common mistakes

- **Counting every token.** Starting the counter at generation start, or counting `start_id` as a thinking token, forces the end sequence one token early on every request.
- **Using `>` instead of `==`.** Forcing only when the count exceeds `budget` allows `budget + 1` thinking tokens, and lets the model emit one thinking token even when `budget = 0`.
- **Forcing only the first token of `forced`.** This bug is hidden when `forced` has length 1. When a cutoff message precedes the closing tag or the closing tag splits into multiple tokens, the span never closes and the downstream parser treats the entire answer as reasoning.
- **Applying `answer_mask` inside the thinking span.** Constraining the thinking span forces the model to reason inside the answer vocabulary, which degrades reasoning quality and desynchronizes stateful grammars before the answer starts.
- **Applying `answer_mask` one token late.** Switching to *answer* on one call and waiting until the next call to mask leaves the first answer token unconstrained.
- **Ignoring the model's own `end_id`.** Counting `end_id` as a regular thinking token keeps the processor in *thinking* during the answer and later injects `forced` into the middle of the answer.
- **Treating `budget = 0` as skipping thinking.** Jumping directly from `start_id` to *answer* leaves the thinking span open instead of emitting `forced`.
- **Reopening the thinking span.** Allowing a `start_id` inside *answer* to re-enter *thinking* grants a second budget and later forces an end sequence into the answer.
- **Blocking `start_id` on the first call.** When `answer_mask` is set, the *start* phase must allow `start_id` alongside `answer_mask`, or constrained requests cannot enter thinking at all.
- **Rewriting the forced token's logit.** Overwriting the forced entry with `0.0` does not change greedy decoding, but it corrupts the token's reported log-probability.

## Complexity

Each call takes $O(V)$ time to construct the masked logits row, $O(1)$ state, and zero model calls. Across a generation, the processor bounds the pre-answer overhead to at most `budget` thinking tokens plus `len(forced)` forced tokens.

## In the wild

Production inference engines implement this exact state machine:

- **vLLM** exposes a per-request `thinking_token_budget` parameter (documented under Reasoning Outputs). Counting begins after the reasoning start string, and once the budget is reached, vLLM forces the reasoning end string configured via `--reasoning-config` (`reasoning_start_str` and `reasoning_end_str`, previously `think_start_str` and `think_end_str`). vLLM's own example prepends a short sentence before `</think>` so the cutoff reads naturally, matching the multi-token `forced` sequence here. For structured outputs, vLLM provides `--structured-outputs-config.enable_in_reasoning=True` to control whether grammar constraints apply during thinking. This problem models the default behavior where the constraint waits until reasoning ends.
- **llama.cpp** provides `--reasoning-budget N` (`-1` for unrestricted, `0` for immediate end, `N > 0` for a token limit) along with `--reasoning-budget-message` for text injected before the closing tag. Its sampler in `common/reasoning-budget.h` uses the same state machine: wait for a start sequence, count until either the budget runs out or a natural end sequence appears, force the end sequence one token at a time, and finish.

This problem simplifies a few real-world details:

- Both `start_id` and `end_id` are single tokens here, whereas real reasoning delimiters are strings that may tokenize into multiple tokens and require sequence matching.
- Thinking can only start on the first generated token and *answer* is final. Some chat templates place the start delimiter inside the prompt so generation begins already in *thinking*, and `llama.cpp` adds an extra state to finish an in-progress multi-byte UTF-8 character before forcing.
- The answer constraint is a static boolean mask rather than a stateful grammar, though deciding when the constraint activates is the same problem.
- Enforcement here is always a hard cutoff. Some model families instead accept a budget hint in the prompt and rely on the model to stop voluntarily, which requires no logits processor.
