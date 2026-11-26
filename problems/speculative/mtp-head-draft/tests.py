"""Tests for mtp-head-draft.

`mtp_draft` is compared with `definition_draft`, the statement in float64 numpy, which
also checks that every argmax it takes has a clear winner (top two logits at least
1e-3 apart), so float32 rounding cannot change a token. `mtp_accept` is compared with
`definition_accept`, a plain loop.

The end-to-end cases run a Medusa-style generation loop on the kit's `LayeredLM`, with
the solution's two functions as the only moving parts: draft k + 1 tokens from the
hidden state of the last accepted position, run the model once over them, accept a
prefix, continue from the hidden state of the last accepted token. Whatever the heads
predict, the output must be the model's own greedy decode. The heads are built here:
random ones, and ones "trained" by least squares on part of the model's own greedy
continuation (their residual branch is zero and their unembedding is fitted), so that
some drafts are right and the loop needs fewer passes than tokens.
"""

import numpy as np

from learnai.judge import case, check
from learnai.kit import LayeredLM

DRAFT = "mtp_draft"
ACCEPT = "mtp_accept"
MARGIN = 1e-3

# The worked example of the statement.
HIDDEN = np.array([1.0, -1.0], dtype=np.float32)
W_OUT = np.array([[1, 0], [0, 1], [-1, -1]], dtype=np.float32)
HEADS = [
    (
        np.zeros((2, 2), dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        np.array([[0, 1], [1, 1], [-1, 0]], dtype=np.float32),
    ),
    (
        np.array([[0, 0], [2, 0]], dtype=np.float32),
        np.array([0, 1], dtype=np.float32),
        np.array([[1, 0], [0, 1], [1, -1]], dtype=np.float32),
    ),
]


def silu(z):
    return z / (1.0 + np.exp(-z))


def top(logits):
    """(argmax, whether the winner is clear)."""
    order = np.sort(logits)
    return int(np.argmax(logits)), bool(order[-1] - order[-2] >= MARGIN)


def definition_draft(hidden, heads, w_out, k):
    """(tokens, decisive) by the statement, in float64."""
    h = np.asarray(hidden, dtype=np.float64)
    token, decisive = top(np.asarray(w_out, dtype=np.float64) @ h)
    tokens = [token]
    for w1, b1, w2 in heads[:k]:
        state = h + silu(np.asarray(w1, dtype=np.float64) @ h + np.asarray(b1, dtype=np.float64))
        token, clear = top(np.asarray(w2, dtype=np.float64) @ state)
        tokens.append(token)
        decisive = decisive and clear
    return tokens, decisive


def definition_accept(draft, target_argmax):
    n = 1
    while n < len(draft) and draft[n] == target_argmax[n - 1]:
        n += 1
    return n


def draft_call(ctx, hidden, heads, w_out, k):
    """Call the solution's mtp_draft; return the tokens as a list of ints."""
    got = np.asarray(ctx.call(DRAFT, hidden, [tuple(head) for head in heads], w_out, k))
    check.that(
        got.dtype.kind in "iu" and got.shape == (k + 1,),
        f"{DRAFT}: must return an integer tensor with one token per drafted position",
        f"k = {k}: expected shape ({k + 1},), got dtype {got.dtype}, shape {got.shape}",
    )
    return [int(t) for t in got]


def check_draft(ctx, hidden, heads, w_out, k, literal=None):
    want, decisive = definition_draft(hidden, heads, w_out, k)
    if not decisive or (literal is not None and literal != want):
        raise RuntimeError("authoring error: a near-tie in the logits, or a wrong literal")
    oracle = ctx.expected(DRAFT, hidden, [tuple(head) for head in heads], w_out, k)
    if [int(t) for t in oracle] != want:
        raise RuntimeError("authoring error: the reference disagrees with the definition")
    got = draft_call(ctx, hidden, heads, w_out, k)
    wrong = [j for j in range(k + 1) if got[j] != want[j]]
    where = (
        "the model's own token (index 0)" if wrong[:1] == [0] else f"head {wrong[0] - 1}"
        if wrong else ""
    )  # fmt: skip
    told = f"{where} gave token {got[wrong[0]]}, expected {want[wrong[0]]}" if wrong else ""
    check.that(
        not wrong,
        f"{DRAFT}: {told}",
        f"call: {DRAFT}(hidden, heads, w_out, k={k}) with D = {hidden.shape[0]}, "
        f"V = {w_out.shape[0]}, {len(heads)} heads\nexpected {want}\ngot {got}",
    )
    return got


def check_accept(ctx, draft, target_argmax, literal=None):
    want = definition_accept(list(draft), list(target_argmax))
    if literal is not None and literal != want:
        raise RuntimeError("authoring error: a wrong literal in the statement")
    args = (np.asarray(draft, dtype=np.int64), np.asarray(target_argmax, dtype=np.int64))
    if int(ctx.expected(ACCEPT, *args)) != want:
        raise RuntimeError("authoring error: the reference disagrees with the definition")
    got = ctx.call(ACCEPT, *args)
    check.that(
        type(got) is int and got == want,
        f"{ACCEPT}: returned {got!r}, expected {want}",
        f"call: {ACCEPT}(draft={list(draft)}, target_argmax={list(target_argmax)})\n"
        f"expected {want}\ngot {got!r}",
    )


def random_heads(ctx, n_heads, d, vocab, scale=1.0):
    """Heads with a residual branch that matters: W1 of order 1, a bias, own unembeddings."""
    return [
        (
            ctx.randn(d, d, scale=scale * d**-0.5),
            ctx.randn(d, scale=0.5 * scale),
            ctx.randn(vocab, d, scale=2.0 * d**-0.5),
        )
        for _ in range(n_heads)
    ]


def decisive_inputs(ctx, n_heads, d, vocab, k, scale=1.0):
    for _ in range(50):
        hidden = ctx.randn(d)
        heads = random_heads(ctx, n_heads, d, vocab, scale)
        w_out = ctx.randn(vocab, d, scale=2.0 * d**-0.5)
        if definition_draft(hidden, heads, w_out, k)[1]:
            return hidden, heads, w_out
    raise RuntimeError("authoring error: no input without a near-tie found")


# -- the model for the end-to-end cases --------------------------------------------------------


class Model:
    """A LayeredLM seen through numpy: hidden states after the final norm, and its head."""

    def __init__(self, ctx, seed):
        self.fw = ctx.oracle_fw
        self.lm = LayeredLM(self.fw, seed=seed)
        d = self.lm.d_model
        # head(I) is the output matrix itself: logits = normalized state @ W.
        self.w_out = np.ascontiguousarray(
            self.fw.numpy(self.lm.head(np.eye(d, dtype=np.float32))).T
        )

    def states(self, tokens, cache=None):
        """Normalized final hidden states (T, D) of `tokens` continuing `cache`."""
        offset = 0 if cache is None else cache.length
        h = self.lm.embed(list(tokens), offset=offset)
        h, cache = self.lm.layers(h, 0, self.lm.n_layers, cache)
        return np.asarray(self.fw.numpy(self.lm.norm(h)), dtype=np.float32), cache

    def greedy(self, prompt, n):
        """(tokens, states): the greedy continuation and the state at every position."""
        seq = list(prompt)
        for _ in range(n):
            states, _ = self.states(seq)
            logits = states[-1].astype(np.float64) @ self.w_out.T.astype(np.float64)
            token, clear = top(logits)
            if not clear:
                raise RuntimeError("authoring error: a near-tie in the model's logits")
            seq.append(token)
        return seq[len(prompt) :], self.states(seq)[0]


def fitted_heads(model, prompt, n_heads, n_train):
    """Heads whose unembedding is fitted to the model's own greedy continuation.

    Head j is right, by construction, at the first `n_train` generated positions, and
    is anybody's guess after them. The residual branch is zero (as a Medusa head is
    initialized), so the head state is the hidden state itself.
    """
    d, vocab = model.w_out.shape[1], model.w_out.shape[0]
    tokens, states = model.greedy(prompt, n_train + n_heads + 2)
    first = len(prompt) - 1  # the state that predicts the first generated token
    heads = []
    for j in range(n_heads):
        x = states[first : first + n_train].astype(np.float64)
        y = np.zeros((n_train, vocab))
        y[np.arange(n_train), tokens[j + 1 : j + 1 + n_train]] = 1.0
        w2 = np.linalg.lstsq(x, y, rcond=None)[0].T
        heads.append(
            (np.zeros((d, d), dtype=np.float32), np.zeros(d, dtype=np.float32),
             w2.astype(np.float32))
        )  # fmt: skip
    return heads


def generate(ctx, model, heads, prompt, k, max_new, solution):
    """The Medusa loop. Returns (tokens, number of model passes after the prompt)."""

    def draft_fn(hidden):
        if solution:
            return draft_call(ctx, hidden, heads, model.w_out, k)
        tokens, decisive = definition_draft(hidden, heads, model.w_out, k)
        if not decisive:
            raise RuntimeError("authoring error: a near-tie in a head's logits")
        return tokens

    def accept_fn(draft, greedy):
        if not solution:
            return definition_accept(draft, greedy)
        n = ctx.call(ACCEPT, np.asarray(draft, dtype=np.int64), np.asarray(greedy, dtype=np.int64))
        check.that(
            type(n) is int and 1 <= n <= len(draft),
            f"{ACCEPT}: must return a Python int between 1 and k + 1",
            f"draft = {draft}, target_argmax = {greedy}, got {n!r}",
        )
        return n

    states, cache = model.states(prompt)
    hidden, out, passes = states[-1], [], 0
    while len(out) < max_new:
        draft = draft_fn(hidden)
        states, longer = model.states(draft, cache)
        passes += 1
        greedy = (states @ model.w_out.T).argmax(axis=-1).tolist()
        n = accept_fn(draft, greedy)
        out += draft[:n]
        hidden = states[n - 1]
        cache = longer.truncate(cache.length + n)
    return out[:max_new], passes


def check_generation(ctx, seed, prompt, heads_for, k, max_new):
    model = Model(ctx, seed)
    heads = heads_for(model)
    want, _ = model.greedy(prompt, max_new)
    expected, expected_passes = generate(ctx, model, heads, prompt, k, max_new, solution=False)
    if expected != want:
        raise RuntimeError("authoring error: the loop does not reproduce greedy decoding")
    got, passes = generate(ctx, model, heads, prompt, k, max_new, solution=True)
    check.that(
        got == want,
        "generation with mtp_draft and mtp_accept does not reproduce the model's greedy decode",
        f"prompt = {prompt}, k = {k}\nexpected {want}\ngot {got}",
    )
    check.that(
        passes == expected_passes,
        f"generation took {passes} model passes, expected {expected_passes} (the number the heads "
        "allow)",
        f"prompt = {prompt}, k = {k}: the same heads need {expected_passes} passes for "
        f"{max_new} tokens when every head token is drafted and verified at its position",
    )
    return expected_passes


# -- cases -------------------------------------------------------------------------------------


@case("the worked example: two heads on a 2-dimensional state, then three verdicts",
      public=True)  # fmt: skip
def worked_example(ctx):
    check_draft(ctx, HIDDEN, HEADS, W_OUT, 2, literal=[0, 1, 1])
    check_draft(ctx, HIDDEN, HEADS, W_OUT, 1, literal=[0, 1])
    check_accept(ctx, [0, 1, 1], [1, 2, 0], literal=2)
    check_accept(ctx, [0, 1, 1], [1, 1, 0], literal=3)
    check_accept(ctx, [0, 1, 1], [2, 1, 0], literal=1)


@case("random heads on a 32-dimensional state: every head reads the base hidden state",
      public=True)  # fmt: skip
def random_draft(ctx):
    # Rejects chained_heads, no_residual, activation_after_add, drops_bias,
    # relu_for_silu, and shared_unembedding.
    for k in (4, 4, 4, 1, 2):
        hidden, heads, w_out = decisive_inputs(ctx, 4, 32, 64, k)
        check_draft(ctx, hidden, heads, w_out, k)


@case("verification is shifted by one: draft[j] is checked against the token after draft[j - 1]",
      public=True)  # fmt: skip
def accept_alignment(ctx):
    # Rejects compares_same_index, counts_all_matches, and forgets_base_token.
    check_accept(ctx, [5, 6, 7, 8, 9], [6, 7, 8, 9, 3])
    check_accept(ctx, [5, 6, 7, 8, 9], [6, 7, 0, 9, 3])
    check_accept(ctx, [5, 6, 7, 8, 9], [0, 7, 8, 9, 3])
    check_accept(ctx, [5, 6, 7, 8, 9], [5, 6, 7, 8, 9])
    check_accept(ctx, [5, 5, 5, 5], [5, 5, 4, 5])
    check_accept(ctx, [3, 4], [4, 4])
    check_accept(ctx, [3, 4], [3, 4])


@case("end to end: heads fitted to the model draft its greedy decode in fewer passes",
      public=True)  # fmt: skip
def fitted_generation(ctx):
    prompt = [0, 17, 42, 5]
    passes = check_generation(
        ctx, 3, prompt, lambda model: fitted_heads(model, prompt, 3, 14), 3, 24
    )
    if not 6 <= passes <= 16:
        raise RuntimeError("authoring error: the fitted heads no longer mix hits and misses")


@case("more heads than k: only the first k are used, in order")
def uses_first_k(ctx):
    # Rejects head_index_off_by_one, which would use heads 1 .. k.
    for k in (1, 2, 3):
        hidden, heads, w_out = decisive_inputs(ctx, 4, 32, 64, 4)
        check_draft(ctx, hidden, heads, w_out, k)


@case("freshly initialized heads: a zero residual branch leaves the hidden state unchanged")
def zero_branch(ctx):
    # W1 = 0 and b1 = 0 give SiLU(0) = 0: the head state is the hidden state, and the
    # tokens differ only through each head's own unembedding.
    d, vocab = 32, 64
    for _ in range(3):
        hidden, heads, w_out = decisive_inputs(ctx, 4, d, vocab, 4)
        zeroed = [(np.zeros_like(w1), np.zeros_like(b1), w2) for w1, b1, w2 in heads]
        if definition_draft(hidden, zeroed, w_out, 4)[1]:
            check_draft(ctx, hidden, zeroed, w_out, 4)
    # Every unembedding equal to the model's head: all k + 1 tokens coincide.
    same = [(np.zeros((d, d), dtype=np.float32), np.zeros(d, dtype=np.float32), w_out)] * 4
    got = check_draft(ctx, hidden, same, w_out, 4)
    if len(set(got)) != 1:
        raise RuntimeError("authoring error: identical heads disagree")


@case("large pre-activations of both signs: SiLU is not ReLU, and the bias matters")
def activation(ctx):
    for scale in (3.0, 6.0):
        hidden, heads, w_out = decisive_inputs(ctx, 4, 32, 64, 4, scale=scale)
        check_draft(ctx, hidden, heads, w_out, 4)
    # A branch that is the bias alone (W1 = 0), negative in every coordinate.
    hidden, heads, w_out = decisive_inputs(ctx, 4, 32, 64, 4)
    biased = [(np.zeros_like(w1), -np.abs(b1) - 0.5, w2) for w1, b1, w2 in heads]
    if not definition_draft(hidden, biased, w_out, 4)[1]:
        raise RuntimeError("authoring error: a near-tie in the logits")
    check_draft(ctx, hidden, biased, w_out, 4)


@case("random drafts and verdicts over a 4-token vocabulary")
def random_accept(ctx):
    rng = ctx.rng
    counts = set()
    for _ in range(60):
        k = int(rng.choice([1, 3, 5]))
        draft = rng.integers(0, 4, size=k + 1)
        target = np.where(rng.random(k + 1) < 0.7, np.roll(draft, -1), rng.integers(0, 4, k + 1))
        check_accept(ctx, draft.tolist(), target.tolist())
        counts.add(definition_accept(draft.tolist(), target.tolist()))
    if len(counts) < 5:
        raise RuntimeError("authoring error: the random verdicts no longer vary")


@case("end to end: random heads are almost never right, and the output is still greedy")
def random_generation(ctx):
    def heads_for(model):
        return random_heads(ctx, 4, model.w_out.shape[1], model.w_out.shape[0])

    check_generation(ctx, 9, [0, 12, 40], heads_for, 4, 12)


@case("end to end with another model, k = 1 and k = 4")
def fitted_generation_other(ctx):
    prompt = [0, 8, 8, 8]
    check_generation(ctx, 4, prompt, lambda model: fitted_heads(model, prompt, 1, 12), 1, 16)
    check_generation(ctx, 4, prompt, lambda model: fitted_heads(model, prompt, 4, 12), 4, 20)
