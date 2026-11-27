## The idea

Because the set of legal outputs is finite and known up front, checking which tokens can follow a prefix is a direct lookup. A trie turns that lookup into a walk from the root. Each node represents one prefix, and its outgoing edges are the tokens that extend that prefix toward at least one choice. Choices that share a beginning share the same path, so each distinct prefix is stored once. Querying takes one dictionary lookup per generated token, regardless of how many choices exist.

Two properties of a node are independent of each other:

- It has **children**, meaning the output can be extended.
- A choice **ends** here, meaning the output can stop.

A leaf has only the second property, and an internal node of a single choice has only the first. When one choice is a prefix of another (such as `yes` and `yes please`), the node has both. Most bugs in this problem come from assuming one property determines the other, such as treating only leaves as complete choices or stopping as soon as a choice ends.

The reference solution records the end marker as an outgoing edge labeled `eos_id`. Because "a choice ends here" and "EOS may come next" are equivalent, the answer for any node is simply its sorted keys.

## Reference solution (Python)

```python
def build_trie(choices, eos_id):
    root = {}
    for choice in choices:
        node = root
        for token in choice:
            node = node.setdefault(token, {})
        node.setdefault(eos_id, {})
    return root


def allowed_next(trie, generated):
    node = trie
    for token in generated:
        node = node.get(token)
        if node is None:
            return []
    return sorted(node)
```

Using `setdefault` makes duplicate choices harmless because inserting the same choice twice walks the same path and finds the `eos_id` edge already present.

A decoding loop that calls `allowed_next` at every step walks from the root each time, which is quadratic in the output length. Choices are short here so that cost is negligible. A stateful decoder avoids re-walking by keeping the current node as its state and following one edge per step, which matches the `start` / `advance` interface used in later problems in this track.

## Common mistakes

- **Allowing EOS at any prefix.** Adding `eos_id` to every response lets the model stop at `neg` or the empty string, producing a prefix instead of a valid choice.
- **Ending choices only at leaves.** With `yes` and `yes please`, the node after `yes` has a child, so EOS is never offered and `yes` becomes unreachable.
- **Allowing only EOS at a complete choice.** Returning `[eos_id]` as soon as a choice ends makes longer extensions like `yes please` unreachable.
- **Returning unsorted IDs.** A Python `dict` iterates in insertion order, and appending `eos_id` at the end puts it last regardless of its numeric value. Callers that merge or binary-search allowed token lists depend on sorted order.
- **Returning duplicate IDs.** Scanning `choices` and collecting each matching choice's next token emits one entry per choice rather than one per unique token.
- **Indexing by position instead of matching the prefix.** Collecting `choice[len(generated)]` across all choices without verifying that each choice starts with `generated` mixes branches. With `[1, 2]` and `[3, 4]`, that bug offers token `4` after `[1]`.

## Complexity

Let $L$ be the total number of tokens across all choices. Building the trie takes $O(L)$ time and memory. Answering a query takes $O(n + k \log k)$ time for $n$ generated tokens and $k$ allowed next tokens. Pre-sorting each node's keys at build time removes the per-query sort, whereas a flat scan over all choices costs $O(L)$ per query.

## In the wild

- **vLLM** accepts `{"structured_outputs": {"choice": [...]}}` (replacing the older `guided_choice` parameter removed in v0.12.0) to guarantee that the output matches one of the listed strings. Rather than building a token trie, it compiles the list into a single grammar rule `root ::= "yes" | "no"` and delegates to its character-level grammar backend.
- That distinction matters in practice. A string often has multiple valid tokenizations (for example, `yes please` can also be tokenized as `y`, `es`, ` please`). A trie over one canonical tokenization per choice forbids alternate tokenizations. That is useful when scoring requires one path per choice, but it can push the model away from a tokenization it prefers. Character-level constraints (as in `dfa-token-mask`) allow every tokenization.
- Token tries are standard in generative retrieval. GENRE (De Cao et al., "Autoregressive Entity Retrieval", ICLR 2021) constrains beam search to a trie of roughly 6 million Wikipedia titles using Hugging Face `generate`'s `prefix_allowed_tokens_fn` hook, which maps the tokens generated so far to the allowed next token IDs just like `allowed_next`.
