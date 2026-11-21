## The idea

An attention layer remembers a prefix as one key and one value per token. Forgetting
the last tokens is dropping entries, so any stored cache serves every prefix of its
sequence, and prefix reuse is "longest common prefix, truncate, continue".

A recurrent layer (a state-space layer, a linear-attention layer) folds each token
into a state of fixed size:

$$
s_t = f(s_{t-1}, x_t).
$$

$s_t$ is as large as $s_1$, and nothing in it says which part came from which token.
There is no inverse of $f$ to apply. A stored state therefore serves exactly one
prefix, the one it has read. Reuse needs a *copy of the state taken at the prefix you
want to return to*, and since you do not know in advance where the next prompt will
part from this one, you take copies at regular positions and at the natural
boundaries.

That gives three decisions.

**Which checkpoint.** The deepest stored key that is a proper prefix of the prompt.
"Shares the most tokens" is the wrong question: an entry that shares six tokens and
then goes on is worth nothing, while an entry of four tokens that the prompt starts
with is worth four. Proper, because the first generated token needs logits, and
logits come out of a call that feeds at least one token.

**Where to save.** At multiples of `interval`, counted from the start of the
sequence so that requests agree on the positions whatever they resumed from, and at
the end of the prompt, which is where a follow-up turn will continue. The states the
model passes through while generating are checkpoints for free: each decode step is
a call, and its state is in hand. During prefill a state exists only at the end of a
call, so the prompt is fed in segments that end where a checkpoint is wanted. With
interval $n$, a prompt that leaves a stored one after $m$ shared tokens resumes at
$\lfloor m / n \rfloor \cdot n$ or later and recomputes fewer than $n$ shared tokens.

**What to keep.** Each checkpoint is a full state. With a budget, the entry worth
keeping is the one requests keep coming back to (the system prompt), so the store is
ordered by use: resuming from a checkpoint refreshes it, saving appends, and eviction
takes from the old end. A dict in insertion order is enough.

Because the model folds tokens in one at a time with the same arithmetic, a run that
resumes from a checkpoint produces the same bits as a fresh run, and the tests
compare tokens and saved states exactly.

## Reference solution (PyTorch)

```python
def generate_with_checkpoints(lm, store, prompt, max_new_tokens, interval, max_checkpoints):
    new_store = dict(store)  # same order: least recently used first

    def save(sequence, state):
        key = tuple(sequence)
        new_store.pop(key, None)
        new_store[key] = state

    best = None
    for key in store:
        usable = len(key) < len(prompt) and tuple(prompt[: len(key)]) == key
        if usable and (best is None or len(key) > len(best)):
            best = key
    state, done = None, 0
    if best is not None:
        state, done = store[best], len(best)
        save(best, state)

    while done < len(prompt):
        stop = min(len(prompt), (done // interval + 1) * interval)
        logits, state = lm(prompt[done:stop], state)
        done = stop
        save(prompt[:done], state)

    sequence, generated = list(prompt), []
    while True:
        token = int(torch.argmax(logits[-1]))
        generated.append(token)
        if len(generated) == max_new_tokens:
            break
        sequence.append(token)
        logits, state = lm([token], state)
        if len(sequence) % interval == 0:
            save(sequence, state)

    while len(new_store) > max_checkpoints:
        del new_store[next(iter(new_store))]
    return generated, new_store
```

The loop over the prompt always runs at least once ($r < P$), so `logits` is
defined. The JAX and MLX solutions differ in the `argmax` line only.

## Common mistakes

- **Treating the store as a KV prefix cache.** Taking the entry with the longest
  common prefix and continuing from its state feeds the model the entry's remaining
  tokens followed by the rest of the prompt. Nothing fails; the tokens are those of
  another prompt.
- **Rounding the shared length up.** "The nearest checkpoint to where the prompts
  part" must be the nearest at or before that point. The one after it has already
  read tokens the prompt does not contain.
- **Using the checkpoint of the whole prompt.** For a repeated prompt the key of all
  $P$ tokens exists. Resuming from it leaves nothing to feed, and feeding the last
  prompt token "to get logits" makes the model read it twice.
- **No checkpoints while generating.** The follow-up turn starts with the prompt and
  the reply. With checkpoints only inside the prompt it resumes from the end of the
  prompt and re-reads the reply on every turn.
- **Only the end of the prompt.** One checkpoint per request serves continuations
  and nothing else: a regenerated answer, an edited message, or a second user with
  the same system prompt finds no usable state.
- **No checkpoint at the end of the prompt.** With multiples of the interval only,
  a follow-up re-reads up to `interval - 1` tokens it need not.
- **Counting the interval from the resume point.** After resuming at length 5 with
  interval 4, the checkpoints belong at 8 and 12, not at 9 and 13. Positions that
  depend on history make two requests with the same prefix save states that neither
  can find.
- **Saving the wrong state under a key.** Storing "the prompt's state" after the
  decode loop saves the state of the prompt plus the reply under the prompt's key,
  and the next request silently inherits the reply.
- **Prefill token by token.** One call per token has a state everywhere and the
  right result, at the cost of $P - r$ calls where a handful would do.
- **Eviction that ignores use.** First-in-first-out evicts the system prompt's
  checkpoint, the oldest and the most useful. Refusing new checkpoints when full
  keeps a store of prefixes nobody sends anymore. Not evicting at all runs out of
  memory, one full state at a time.

## Complexity

With $r$ tokens resumed: $P - r + g - 1$ tokens through the model, in
$s + g - 1$ calls where $s \le \lceil (P - r) / n \rceil + 1$. A request saves at
most $\lfloor (P + g - 1) / n \rfloor + 1$ checkpoints, each of `state.nbytes`
bytes, whatever the sequence length. The choice of $n$ trades memory for
recomputation: the expected waste on a divergent prompt is about $n / 2$ tokens,
and the store grows as $1 / n$.

## In the wild

- **Why the state cannot be rolled back** is stated in "Hybrid Models Meet SGLang:
  More than Full Attention" (PyTorch blog, December 2025): state-space states are
  updated in place, so a request's state cannot be rolled back to represent its
  prefixes; a state is orders of magnitude larger than one token's keys and values;
  and reuse is all or nothing. SGLang's answer, per that post, is a radix tree
  (`MambaRadixCache`) whose match step returns the deepest node that holds a state
  and whose key is a prefix of the input, copying the state out; whose insert step
  forks a checkpoint of the request's state after chunked prefill or decoding; and
  which evicts states and KV separately by recency, states from any node. That is
  this problem's lookup rule, its "save where a call ends", and its LRU.
- **llama.cpp** added the same idea to its server in the pull request "implement
  context checkpointing for hybrid and recurrent models" (ggml-org/llama.cpp
  #16382), which generalized its sliding-window-attention checkpoints to recurrent
  and hybrid models. The server's README documents the two knobs:
  `--ctx-checkpoints N`, the maximum number of context checkpoints per slot
  (default 32), and `--checkpoint-min-step N`, the minimum spacing between
  checkpoints in tokens (default 8192). They correspond to `max_checkpoints` and,
  loosely, `interval`.
- **The layers** in question are those of "Mamba: Linear-Time Sequence Modeling
  with Selective State Spaces" (arXiv 2312.00752) and of "Gated Delta Networks:
  Improving Mamba2 with Delta Rule" (arXiv 2412.06464), the linear-attention layer
  that the SGLang post serves in Qwen3-Next and that the Qwen3.5 checkpoints ship
  in three layers out of four (the `gated-delta-step` and `gated-delta-layer`
  problems build it).
- **What this problem simplifies.** The model here is purely recurrent. A hybrid
  model also has attention layers whose KV cache *can* be truncated, and an engine
  must keep the two in step: it restores the recurrent state from a checkpoint and
  cuts the KV to the same length. The store here is a flat dict searched linearly; a
  radix tree finds the deepest usable node in time proportional to the prompt. The
  positions where real engines save differ from this problem's fixed interval
  (after prefill chunks and at the end of processing in the sources above), and the
  exact eviction rule here (strict least-recently-used over one list) is this
  problem's choice, not a description of either engine.
