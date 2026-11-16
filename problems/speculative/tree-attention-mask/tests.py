"""Tests for tree-attention-mask.

Three kinds of checks:

- numeric: `tree_inputs` against `definition_inputs` (a loop over nodes that walks up
  the parent pointers) and `verify_tree` against `definition_verify` (a walk down the
  tree), both plain Python written from the statement;
- property (the point of the construction): a small decoder (the kit's
  `decoder_block_ref`: RoPE positions, grouped-query attention) is run once over the
  prefix followed by the flattened tree, with the solution's mask and position ids. The
  logits at every node must equal the logits of running the prefix and that node's
  root path alone, as an ordinary causal sequence, and the prefix rows must be
  unchanged by the tree behind them;
- end to end: the greedy tokens of that single pass are handed to `verify_tree`, and
  the accepted tokens plus the next token must be the model's own greedy decode.
"""

import numpy as np

from learnai.judge import case, check
from learnai.kit import DecoderConfig, decoder_block_ref, decoder_params, layer_params, ops

INPUTS = "tree_inputs"
VERIFY = "verify_tree"
CFG = DecoderConfig(vocab_size=32, d_model=16, n_layers=2, n_heads=2, n_kv_heads=1, d_mlp=24)

# The worked example of the statement.
PARENTS = [-1, -1, 0, 0, 1, 2]
TOKENS = [7, 3, 4, 9, 8, 6]


# -- definitions ---------------------------------------------------------------------------


def root_path(parents, node):
    """Node indices from the top of the tree down to `node`."""
    path = []
    while node >= 0:
        path.append(node)
        node = parents[node]
    return path[::-1]


def definition_inputs(parents, prefix_len):
    n = len(parents)
    size = prefix_len + n
    mask = np.zeros((size, size), dtype=bool)
    positions = np.zeros(size, dtype=np.int64)
    for row in range(prefix_len):
        mask[row, : row + 1] = True
        positions[row] = row
    for node in range(n):
        path = root_path(parents, node)
        mask[prefix_len + node, :prefix_len] = True
        mask[prefix_len + node, [prefix_len + j for j in path]] = True
        positions[prefix_len + node] = prefix_len + len(path) - 1
    return mask, positions


def definition_verify(parents, tokens, target_argmax):
    path, current, want = [], -1, int(target_argmax[0])
    while True:
        children = [i for i in range(len(parents)) if parents[i] == current and tokens[i] == want]
        if not children:
            return path, want
        current = children[0]
        path.append(current)
        want = int(target_argmax[current + 1])


# -- calling the solution --------------------------------------------------------------------


def inputs(ctx, parents, prefix_len):
    """Call tree_inputs; check types and shapes; return (mask, position_ids) as numpy."""
    result = ctx.call(INPUTS, np.asarray(parents, dtype=np.int64), prefix_len)
    check.that(
        isinstance(result, (tuple, list)) and len(result) == 2,
        f"{INPUTS}: expected a (mask, position_ids) pair, got {type(result).__name__}",
    )
    mask, positions = np.asarray(result[0]), np.asarray(result[1])
    size = prefix_len + len(parents)
    check.that(
        mask.dtype == np.bool_,
        f"{INPUTS}: the mask has dtype {mask.dtype}; it must be boolean (True = may attend)",
    )
    check.shape(mask, (size, size), label=f"{INPUTS}: mask")
    check.that(
        positions.dtype.kind in "iu",
        f"{INPUTS}: position_ids has dtype {positions.dtype}; it must be an integer tensor",
    )
    check.shape(positions, (size,), label=f"{INPUTS}: position_ids")
    return mask, positions


def show(mask):
    return "\n".join("    " + " ".join("1" if v else "." for v in row) for row in mask)


def check_inputs(ctx, parents, prefix_len, literal=None):
    want_mask, want_pos = definition_inputs(parents, prefix_len)
    if literal is not None and (
        show(want_mask) != show(np.array(literal[0], dtype=bool)) or want_pos.tolist() != literal[1]
    ):
        raise RuntimeError("authoring error: a wrong literal in the statement")
    oracle = ctx.expected(INPUTS, np.asarray(parents, dtype=np.int64), prefix_len)
    if not (np.array_equal(oracle[0], want_mask) and np.array_equal(oracle[1], want_pos)):
        raise RuntimeError("authoring error: the reference disagrees with the definition")
    mask, positions = inputs(ctx, parents, prefix_len)
    detail = (
        f"call: {INPUTS}(parents={list(parents)}, prefix_len={prefix_len})\n"
        f"expected mask (rows attend to columns marked 1):\n{show(want_mask)}\n"
        f"got mask:\n{show(mask)}\n"
        f"expected position_ids = {want_pos.tolist()}\ngot position_ids = {positions.tolist()}"
    )
    wrong = np.argwhere(mask != want_mask)
    if len(wrong):
        row, col = (int(v) for v in wrong[0])
        where = "a prefix row" if row < prefix_len else f"the row of node {row - prefix_len}"
        what = "attends to" if mask[row, col] else "does not attend to"
        target = f"prefix column {col}" if col < prefix_len else f"node {col - prefix_len}"
        check.that(False, f"{INPUTS}: {where} {what} {target} ({len(wrong)} wrong entries)", detail)
    wrong = np.flatnonzero(positions != want_pos)
    check.that(
        len(wrong) == 0,
        f"{INPUTS}: wrong position id at flat index {wrong[:1].tolist()} "
        f"({len(wrong)} wrong entries)",
        detail,
    )


def check_verify(ctx, parents, tokens, target_argmax, literal=None):
    want = definition_verify(parents, tokens, target_argmax)
    if literal is not None and literal != want:
        raise RuntimeError("authoring error: a wrong literal in the statement")
    args = [np.asarray(a, dtype=np.int64) for a in (parents, tokens, target_argmax)]
    oracle = ctx.expected(VERIFY, *args)
    if (list(oracle[0]), int(oracle[1])) != want:
        raise RuntimeError("authoring error: the reference disagrees with the definition")
    result = ctx.call(VERIFY, *args)
    check.that(
        isinstance(result, (tuple, list)) and len(result) == 2,
        f"{VERIFY}: expected a (path, next_token) pair, got {type(result).__name__}",
    )
    path, next_token = result
    detail = (
        f"call: {VERIFY}(parents, tokens, target_argmax)\n  parents = {list(parents)}\n"
        f"  tokens = {list(tokens)}\n  target_argmax = {list(target_argmax)}\n"
        f"expected path = {want[0]}, next_token = {want[1]}\n"
        f"got path = {path!r}, next_token = {next_token!r}"
    )
    check.that(
        isinstance(path, list) and all(type(i) is int for i in path) and type(next_token) is int,
        f"{VERIFY}: must return a list of Python ints and a Python int",
        detail,
    )
    check.that(path == want[0], f"{VERIFY}: wrong accepted path", detail)
    check.that(next_token == want[1], f"{VERIFY}: wrong next token", detail)
    return path, next_token


# -- the model the tree is verified with -------------------------------------------------------


def forward(params, tokens, mask, positions):
    """Logits (T, V) of the small decoder for `tokens` under `mask` and `positions`."""
    embed = params["model.embed_tokens.weight"]
    x = embed[np.asarray(tokens)][None]
    for i in range(CFG.n_layers):
        x = decoder_block_ref(CFG, layer_params(params, i), x, mask, np.asarray(positions))
    x = ops.rms_norm(x, params["model.norm.weight"], CFG.norm_eps)
    return (x @ embed.T)[0].astype(np.float32)


def linear(params, tokens):
    """The ordinary causal forward pass over one sequence."""
    size = len(tokens)
    return forward(params, tokens, np.tril(np.ones((size, size), dtype=bool)), np.arange(size))


def check_equivalence(ctx, params, prefix, parents, tokens):
    """One pass over prefix + tree with the solution's inputs equals every path alone."""
    p = len(prefix)
    mask, positions = inputs(ctx, parents, p)
    tree_logits = forward(params, [*prefix, *tokens], mask, positions)
    alone = linear(params, prefix)
    err = float(np.abs(tree_logits[:p] - alone).max())
    check.that(
        err <= 1e-4,
        f"{INPUTS}: the prefix's logits change when the tree is appended "
        f"(error on the order of 1e{int(np.floor(np.log10(max(err, 1e-30))))})",
        f"prefix = {prefix}\nparents = {list(parents)}\nmask:\n{show(mask)}\n"
        f"position_ids = {positions.tolist()}",
    )
    for node in range(len(parents)):
        path = root_path(parents, node)
        alone = linear(params, [*prefix, *[tokens[j] for j in path]])[-1]
        err = float(np.abs(tree_logits[p + node] - alone).max())
        check.that(
            err <= 1e-4,
            f"{INPUTS}: the logits at node {node} differ from running its root path alone "
            f"(error on the order of 1e{int(np.floor(np.log10(max(err, 1e-30))))})",
            f"prefix = {prefix}\nparents = {list(parents)}\ntokens = {list(tokens)}\n"
            f"node {node} has root path {path}: its row must reproduce the sequence "
            f"{[*prefix, *[tokens[j] for j in path]]}\nmask:\n{show(mask)}\n"
            f"position_ids = {positions.tolist()}",
        )
    return tree_logits


def random_tree(rng, n, branch=0.5):
    """Parents in topological order: each node hangs under an earlier node or the prefix."""
    parents = [-1]
    for i in range(1, n):
        parents.append(int(rng.integers(0, i)) if rng.random() < 1 - branch / 3 else -1)
    return parents


def distinct_sibling_tokens(rng, parents, vocab):
    """Random tokens such that no two children of one parent share a token."""
    tokens, used = [], {}
    for parent in parents:
        taken = used.setdefault(parent, set())
        token = int(rng.integers(0, vocab))
        while token in taken:
            token = int(rng.integers(0, vocab))
        taken.add(token)
        tokens.append(token)
    return tokens


def greedy_tree(rng, params, prefix, depth, decoys):
    """A tree whose every level holds the model's greedy token among decoys.

    Returns (parents, tokens, greedy continuation of the prefix, `depth + 1` tokens).
    """
    seq, greedy = list(prefix), []
    for _ in range(depth + 1):
        greedy.append(int(linear(params, seq)[-1].argmax()))
        seq.append(greedy[-1])
    parents, tokens, on_path = [], [], -1
    for level in range(depth):
        options = [greedy[level]]
        while len(options) < decoys + 1:
            token = int(rng.integers(0, CFG.vocab_size))
            if token not in options:
                options.append(token)
        order = rng.permutation(len(options))
        first = len(parents)
        for slot in order:
            parents.append(on_path)
            tokens.append(options[slot])
        # A decoy gets a child of its own, carrying the token that would be right one
        # level further down the greedy path: it must not be accepted.
        decoy = first + int(np.flatnonzero(order != 0)[0])
        parents.append(decoy)
        tokens.append(greedy[level + 1])
        on_path = first + int(np.flatnonzero(order == 0)[0])
    return parents, tokens, greedy


# -- cases -------------------------------------------------------------------------------------


@case("the worked example: a six-node tree behind a two-token prefix", public=True)
def worked_example(ctx):
    mask = [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 0, 1, 0],
        [1, 1, 1, 0, 1, 0, 0, 1],
    ]
    check_inputs(ctx, PARENTS, 2, literal=(mask, [0, 1, 2, 2, 3, 3, 3, 4]))
    check_verify(ctx, PARENTS, TOKENS, [7, 9, 8, 6, 2, 5, 1], literal=([0, 3], 2))
    check_verify(ctx, PARENTS, TOKENS, [7, 4, 8, 6, 2, 5, 1], literal=([0, 2, 5], 1))
    check_verify(ctx, PARENTS, TOKENS, [5, 9, 8, 6, 2, 5, 1], literal=([], 5))


@case("a chain is an ordinary causal sequence; a star puts every node at the same position",
      public=True)  # fmt: skip
def chain_and_star(ctx):
    check_inputs(ctx, [-1, 0, 1, 2, 3], 3)
    check_inputs(ctx, [-1, -1, -1, -1], 3)
    check_inputs(ctx, [-1], 1)
    check_inputs(ctx, [-1, 0, 0, 1, 1, 2, 2], 1)


@case("one pass over the tree gives every node the logits of its own path", public=True)
def equivalence(ctx):
    # Rejects siblings_visible, positions_by_flat_index, parent_only, no_self_attention,
    # depth_from_one, and prefix_sees_everything by what they do to the model's output.
    params = decoder_params(CFG, seed=11)
    check_equivalence(ctx, params, [1, 20], PARENTS, TOKENS)
    check_equivalence(
        ctx, params, [5, 6, 7, 8], [-1, 0, 1, 2, 2, 2, 0, -1], [9, 3, 3, 4, 12, 30, 9, 2]
    )


@case("verification walks down from the prefix: one child per level, the first mismatch ends it",
      public=True)  # fmt: skip
def verify_walk(ctx):
    # Rejects first_child_only, argmax_misaligned, and next_token_is_last_draft.
    parents = [-1, -1, -1, 1, 1, 4, 4, 0]
    tokens = [10, 11, 12, 20, 21, 30, 31, 22]
    #          after: prefix, n0, n1, n2, n3, n4, n5, n6, n7
    check_verify(ctx, parents, tokens, [11, 22, 21, 0, 0, 31, 0, 40, 0])  # down to node 6
    check_verify(ctx, parents, tokens, [11, 22, 21, 0, 0, 32, 0, 40, 0])  # stops at node 4
    check_verify(ctx, parents, tokens, [12, 22, 21, 50, 0, 31, 0, 40, 0])  # a leaf at the top
    check_verify(ctx, parents, tokens, [13, 22, 21, 0, 0, 31, 0, 40, 0])  # nothing accepted


@case("a node under a rejected parent is not accepted, although its own token is right")
def no_orphans(ctx):
    # Rejects accepts_without_ancestors. Node 2 (under node 1) carries the token the
    # target emits after node 1, and node 1 itself is wrong.
    check_verify(ctx, [-1, -1, 1, 2, 0], [5, 6, 7, 8, 9], [5, 3, 7, 8, 4, 0])
    check_verify(ctx, [-1, 0, 1, 2], [5, 6, 7, 8], [4, 6, 7, 8, 9])
    check_verify(ctx, [-1, 0, 1, 2], [5, 6, 7, 8], [5, 6, 0, 8, 9])
    check_verify(ctx, [-1, 0, 1, 2], [5, 6, 7, 8], [5, 6, 7, 8, 9])


@case("random trees: mask and position ids for several sizes and prefix lengths")
def random_inputs(ctx):
    # A short list of sizes: JAX compiles once per shape.
    for n, prefix_len in [(12, 1), (12, 5), (20, 5), (20, 9), (12, 9)]:
        for branch in (0.2, 0.9, 1.8):
            check_inputs(ctx, random_tree(ctx.rng, n, branch), prefix_len)


@case("random trees: every node's logits equal those of its path alone")
def equivalence_random(ctx):
    params = decoder_params(CFG, seed=5)

    def prop(rng):
        prefix = [int(t) for t in rng.integers(0, CFG.vocab_size, size=5)]
        parents = random_tree(rng, 12, float(rng.choice([0.3, 1.0, 1.8])))
        tokens = [int(t) for t in rng.integers(0, CFG.vocab_size, size=12)]
        check_equivalence(ctx, params, prefix, parents, tokens)

    check.forall(prop, trials=6)


@case("random trees and random target tokens: the accepted path and the next token")
def random_verify(ctx):
    rng = ctx.rng
    lengths = []
    for _ in range(40):
        n = int(rng.choice([6, 12, 20]))
        parents = random_tree(rng, n, float(rng.choice([0.3, 1.0, 1.8])))
        tokens = distinct_sibling_tokens(rng, parents, 32)
        # The target usually continues with the token of one of the children, so that
        # walks of every length occur; otherwise with a token no child carries.
        target = []
        for node in range(-1, n):
            children = [tokens[i] for i in range(n) if parents[i] == node]
            if children and rng.random() < 0.75:
                target.append(int(rng.choice(children)))
            else:
                target.append(32 + int(rng.integers(0, 4)))
        path, _ = check_verify(ctx, parents, tokens, target)
        lengths.append(len(path))
    if not (lengths.count(0) >= 3 and max(lengths) >= 3):
        raise RuntimeError("authoring error: the random trees no longer mix path lengths")


@case("end to end: one masked pass, then verification, reproduces the model's greedy decode")
def end_to_end(ctx):
    for seed, prefix, depth in [(3, [4, 9, 2], 3), (8, [1, 30, 7, 7, 15], 4)]:
        params = decoder_params(CFG, seed=seed)
        parents, tokens, greedy = greedy_tree(ctx.rng, params, prefix, depth, decoys=2)
        logits = check_equivalence(ctx, params, prefix, parents, tokens)
        gaps = np.sort(logits, axis=-1)
        if (gaps[:, -1] - gaps[:, -2]).min() < 1e-3:
            raise RuntimeError("authoring error: a near-tie in the model's logits")
        target_argmax = logits[len(prefix) - 1 :].argmax(axis=-1).tolist()
        path, next_token = check_verify(ctx, parents, tokens, target_argmax)
        emitted = [*[tokens[i] for i in path], next_token]
        if emitted != greedy:
            raise RuntimeError("authoring error: the tree does not contain the greedy path")
    # The same tree with the greedy path cut after two levels: the walk stops there.
    cut = [i for i in range(len(parents)) if len(root_path(parents, i)) <= 2]
    remap = {old: new for new, old in enumerate(cut)}
    parents2 = [remap.get(parents[i], -1) for i in cut]
    tokens2 = [tokens[i] for i in cut]
    logits = check_equivalence(ctx, params, prefix, parents2, tokens2)
    path, next_token = check_verify(
        ctx, parents2, tokens2, logits[len(prefix) - 1 :].argmax(axis=-1).tolist()
    )
    if [*[tokens2[i] for i in path], next_token] != greedy[:3]:
        raise RuntimeError("authoring error: the cut tree is not accepted to its leaf")
