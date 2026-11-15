## The idea

A logits processor is a function from "what has been generated" and "what the model
wants next" to "what may come next". This one needs very little of the history: a
phase and one counter. The four phases form a line with one shortcut,

```
start ──start_id──> thinking ──budget used up──> forcing ──end sequence done──> answer
  │                    │                                                          ▲
  └── any other token ─┼──────────────── end_id (the model closes) ───────────────┘
```

and each has one rule for the logits: *start* and *answer* apply the answer mask
(the start phase adds `start_id`, so that a constrained request can still think),
*thinking* applies nothing, and *forcing* leaves a single token.

Two decisions carry the correctness.

**What counts.** The budget counts thinking tokens: the tokens after `start_id`
that the model chose freely inside the span. Not the start token, not the tokens of
the end sequence, not an `end_id` the model produced itself. With that definition
"at most `budget` thinking tokens" holds for every run, including `budget = 0`,
where the end sequence follows the start token directly.

**When to check.** The call that receives the token which uses up the budget is
the call that must already return the forced row: the check "thinking and
$u =$ budget" belongs after the transition, in the same call. A check before the
transition, or with `>`, is late by one token.

Forcing means masking, not rewriting. Every other token is set to `-inf` and the
forced token keeps its logit, so any sampler that follows (greedy, temperature,
top-p) has exactly one token to choose and needs no special case.

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

The state is a tuple of a string and an int, so the function is pure and the same
text works in JAX and MLX with `jnp.where` and `mx.where`. Both the transition and
the row are a handful of comparisons: the processor costs nothing next to a forward
pass, and it never calls the model.

## Common mistakes

- **Counting every token.** A counter that starts when generation starts, or that
  counts the start token, forces the end sequence one token early on every request.
- **`>` for `==`.** Forcing when the count *exceeds* the budget lets `budget + 1`
  thinking tokens through, and with a budget of 0 the model thinks once before it
  is stopped.
- **Forcing only the first token of the end sequence.** With a one-token marker
  this is invisible. With a message in front of the marker, or a marker that the
  tokenizer splits, the processor declares the span closed while the closing tag
  has not been produced; the output parser then treats the whole answer as
  reasoning.
- **The answer mask inside the span.** The constraint describes the answer.
  Applied while thinking, it makes the model reason in the answer's alphabet,
  which destroys the reasoning and, for a grammar with state, desynchronizes the
  grammar before the answer begins.
- **The mask one token late.** Switching to the answer phase in one call and
  masking from the next leaves the first answer token unconstrained.
- **Ignoring the model's own close.** If `end_id` from the model is counted as one
  more thinking token, the processor is still "thinking" during the answer and
  later forces the end sequence into the middle of it.
- **A budget of 0 as "skip thinking".** Going straight to the answer after the
  start token leaves an open span: the end sequence still has to be produced.
- **Reopening.** Treating a later `start_id` as a new span gives the model a fresh
  budget, and then forces an end sequence into the answer.
- **Blocking the start token.** With an answer mask, the first row must allow
  `start_id` as well, or constrained requests cannot think at all.
- **Rewriting the forced logit.** Setting the forced token to 0 changes nothing
  for a greedy sampler and changes the reported log-probability of the token.

## Complexity

$O(V)$ per call for the row, $O(1)$ state, no model calls. The cost that matters
is the one it bounds: at most `budget` thinking tokens plus `len(forced)` tokens
before the answer starts.

## In the wild

What is established, from the projects' own documentation:

- **vLLM** documents a thinking budget on its "Reasoning Outputs" page:
  `thinking_token_budget` is a per-request sampling parameter; token counting
  starts from the reasoning start string, and once the count reaches the budget
  vLLM forces the model to produce the reasoning end string. The two strings come
  from the reasoning parser or from `--reasoning-config` (`reasoning_start_str`,
  `reasoning_end_str`; older releases call them `think_start_str` and
  `think_end_str`). The page's own example makes the end string a sentence followed
  by `</think>`, so that the cutoff reads naturally: that is the multi-token end
  sequence of this problem. The page does not say what a budget of 0 does.
- **vLLM's structured outputs** page has the switch for the other half:
  `--structured-outputs-config.enable_in_reasoning=True`, which, in its words,
  determines when constraints apply. This problem models the setting in which the
  constraint waits for the end of reasoning.
- **llama.cpp**'s server README lists `--reasoning-budget N` ("-1 for
  unrestricted, 0 for immediate end, N>0 for token budget") and
  `--reasoning-budget-message`, a message injected before the end-of-thinking tag
  when the budget is exhausted. Its sampler (`common/reasoning-budget.h`) is a
  state machine with the same shape as the one here: idle until a start sequence,
  counting until the budget runs out or an end sequence appears naturally, forcing
  a given token sequence one token at a time, then done.

What this problem simplifies:

- The start marker is one token and the natural end marker is one token; real
  markers are strings that may tokenize into several tokens, so engines match token
  sequences.
- A span can only open with the first generated token, and the answer phase is
  final. Chat templates differ here (some put the start marker in the prompt, so
  generation begins inside the span), and llama.cpp's sampler has an extra state
  that waits for a multi-byte UTF-8 character to complete before it starts forcing.
- The answer constraint is a fixed mask. A real constraint has state (a grammar
  position), and the processor's job is then to decide *when* its mask applies and
  when its state advances, which is the same decision.
- A hard cutoff is the only policy here. Some model families instead take the
  budget as a prompt-level hint and are expected to respect it themselves; that is
  not enforcement and needs no processor.
