## The idea

The set of legal outputs is finite and known, so the constraint is a lookup: given the
tokens so far, which tokens continue at least one choice? A trie makes that lookup a
walk. Every node stands for one prefix; its outgoing edges are the tokens that extend
the prefix toward some choice. Choices that share a beginning share a path, so each
distinct prefix is stored once, and a query costs one dict lookup per generated token,
however many choices there are.

Two facts about a node are independent of each other:

- it has **children**: the output can be extended;
- a choice **ends** here: the output can stop.

A leaf has only the second, an inner node of a single choice only the first, and when
one choice is a prefix of another (`yes`, `yes please`) the node has both. Most bugs
in this problem come from deriving one fact from the other ("no children, so it is an
end", "it is an end, so nothing follows").

The reference stores the end marker as an edge labeled EOS. "A choice ends here" and
"EOS may come next" are the same statement, so the answer to a query is just the
sorted keys of the node.

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

`setdefault` makes duplicates harmless: inserting a choice twice walks the same path
and finds the EOS edge already there.

A decoding loop that calls `allowed_next` at every step walks from the root each time,
which is quadratic in the output length. Choices are short, so it does not matter
here; a loop that cares keeps the current node as its state and follows one edge per
step, which is the `start` / `advance` interface of the later problems in this track.

## Common mistakes

- **EOS at any prefix.** Adding EOS to every answer lets the model stop at `neg` or at
  the empty string. The output is then a prefix of a choice, not a choice.
- **A choice ends only at a leaf.** With `yes` and `yes please`, the node after `yes`
  has a child, so EOS is never offered and `yes` cannot be selected.
- **A complete choice allows only EOS.** The mirror image: returning `[eos_id]` as
  soon as a choice ends makes `yes please` unreachable.
- **Unsorted ids.** A dict lists keys in insertion order, which is the order of the
  `choices` argument, and appending EOS at the end puts it last whatever its id.
  Callers that merge allowed sets or binary-search them rely on the order.
- **Duplicated ids.** Scanning the choices and collecting each matching choice's next
  token yields one entry per choice, not per token.
- **Position instead of path.** Collecting `choice[len(generated)]` over all choices
  without checking that the choice starts with `generated` mixes branches: with
  `[1, 2]` and `[3, 4]`, token `4` is offered after `[1]`.

## Complexity

Let $L$ be the total number of tokens in all choices. Building takes $O(L)$ time and
memory. A query takes $O(n + k \log k)$ for $n$ generated tokens and an answer of $k$
ids; storing each node's sorted answer at build time removes the sort. A flat scan
over the choices costs $O(L)$ per query instead.

## In the wild

- **vLLM** accepts `{"structured_outputs": {"choice": [...]}}` (the older
  `guided_choice` parameter was removed in v0.12.0) and guarantees that the output
  is exactly one of the strings. It does not build a token trie: it turns the list
  into a one-rule grammar, `root ::= "yes" | "no"`, and hands that to its grammar
  backend, which works on characters.
- That difference matters. A string usually has many tokenizations (`yes please` is
  also `y`, `es`, ` please`, ...). A trie over one canonical tokenization per choice,
  as in this problem, forbids the others. This is usually what you want for scoring
  (one path per choice) and occasionally a problem (the model is pushed away from a
  tokenization it prefers). A character-level constraint, as in `dfa-token-mask`,
  admits every tokenization.
- The token trie is the standard tool in generative retrieval. GENRE (De Cao et al.,
  "Autoregressive Entity Retrieval", ICLR 2021) constrains beam search to a trie of
  all Wikipedia titles, about 6 million leaves, through the `prefix_allowed_tokens_fn`
  hook of Hugging Face `generate`: a function from the tokens so far to the allowed
  ids, like `allowed_next`.
