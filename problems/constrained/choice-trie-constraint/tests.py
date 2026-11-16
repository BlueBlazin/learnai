"""Tests for choice-trie-constraint.

The worked example is compared with literal ids; everything else with the oracle, on
every prefix of every choice and on sequences that leave the trie. The trie itself is
opaque: the tests only hand it back to `allowed_next`.
"""

from learnai.judge import case, check
from learnai.kit import ToyVocab

# The worked example of the statement. 7 = "yes", 12 = " please", 9 = "no", 4 = "t",
# 3 = " sure"; EOS is 10.
CHOICES = [[7], [7, 12], [9], [9, 4, 3]]
EOS = 10


def ask(ctx, trie, generated, label=None):
    """Call the solution's allowed_next and check the type of the answer."""
    label = label or f"allowed_next(trie, {generated})"
    got = ctx.call("allowed_next", trie, list(generated))
    check.that(
        isinstance(got, list) and all(type(token) is int for token in got),
        f"{label}: must return a list of Python ints",
        f"got {got!r}",
    )
    return got


def check_everywhere(ctx, choices, eos_id, extra=()):
    """Compare with the oracle on every prefix of every choice, and on `extra`."""
    trie = ctx.call("build_trie", [list(choice) for choice in choices], eos_id)
    oracle_trie = ctx.expected("build_trie", [list(choice) for choice in choices], eos_id)
    prefixes = {tuple(choice[:n]) for choice in choices for n in range(len(choice) + 1)}
    for generated in [*sorted(prefixes), *extra]:
        generated = list(generated)
        expected = ctx.expected("allowed_next", oracle_trie, generated)
        label = f"allowed_next for choices {choices}, eos_id {eos_id}, generated {generated}"
        check.equal(ask(ctx, trie, generated, label), expected, label=label)
    return trie


@case("the worked example: yes / yes please / no / not sure", public=True)
def worked_example(ctx):
    trie = ctx.call("build_trie", [list(choice) for choice in CHOICES], EOS)
    for generated, expected in [
        ([], [7, 9]),
        ([7], [10, 12]),
        ([7, 12], [10]),
        ([9], [4, 10]),
        ([9, 4], [3]),
        ([9, 4, 3], [10]),
        ([12], []),
        ([7, 9], []),
    ]:
        label = f"allowed_next(trie, {generated})"
        check.equal(ask(ctx, trie, generated), expected, label=label)


@case("a single choice: EOS only after its last token", public=True)
def single_choice(ctx):
    # Rejects eos_any_prefix.
    check_everywhere(ctx, [[5, 6, 5, 8]], 1)
    check_everywhere(ctx, [[3]], 0)


@case("ids come back sorted, wherever EOS falls among them", public=True)
def sorted_ids(ctx):
    # Rejects insertion_order: choices listed in decreasing order, EOS below, between,
    # and above the other ids.
    choices = [[40, 30], [40, 20], [40], [35], [25, 1], [25]]
    for eos_id in (0, 28, 99):
        check_everywhere(ctx, choices, eos_id)


@case("a sequence that is not a prefix of any choice allows nothing", public=True)
def off_the_trie(ctx):
    choices = [[1, 2, 3], [1, 4], [5]]
    extra = [[2], [3], [1, 3], [1, 2, 4], [1, 2, 3, 3], [5, 5], [1, 4, 2], [9], [4, 1], [5, 1, 2]]
    check_everywhere(ctx, choices, 0, extra)


@case("choices that are prefixes of each other: a, ab, abc")
def nested_choices(ctx):
    # Rejects eos_only_at_leaves and terminal_blocks_longer, in both insertion orders.
    for choices in ([[1], [1, 2], [1, 2, 3]], [[1, 2, 3], [1, 2], [1]], [[4, 4, 4], [4], [4, 4]]):
        check_everywhere(ctx, choices, 9)
    check_everywhere(ctx, [[1, 2, 3, 4], [1, 2]], 0)
    check_everywhere(ctx, [[1, 2], [1, 2, 3, 4], [1, 2, 3, 5]], 7)


@case("several choices share their next token; a choice is listed twice")
def shared_tokens(ctx):
    # Rejects duplicate_ids.
    check_everywhere(ctx, [[1, 2, 3], [1, 2, 4], [1, 5]], 0)
    check_everywhere(ctx, [[6, 7], [6, 7], [6]], 2)
    check_everywhere(ctx, [[3], [3], [3]], 1)


@case("the same token at the same position on different branches")
def branches_do_not_mix(ctx):
    # Rejects position_only: after [1] only 2 may follow, not the 4 of the other branch.
    check_everywhere(ctx, [[1, 2], [3, 4]], 0)
    check_everywhere(ctx, [[1, 2, 5], [3, 2, 6], [3, 4]], 9, extra=[[1, 4], [3, 2, 5]])
    check_everywhere(ctx, [[8, 1], [2, 8, 3], [2]], 5)


@case("choices tokenized with a real vocabulary: true, false, null, and friends")
def toy_vocab_choices(ctx):
    for seed in (0, 3):
        vocab = ToyVocab(size=64, seed=seed)
        texts = ["true", "false", "null", "tr", "fal", "truest", '{"a": true}', '{"a": false}']
        texts += ["-1", "-10", "-1.5", "1", "é", "€", "€€", "😀"]
        choices = [vocab.encode(text) for text in texts]
        check_everywhere(ctx, choices, vocab.eos_id)


@case("random choice sets over a small alphabet, with random off-trie queries")
def random_choices(ctx):
    rng = ctx.rng
    for _ in range(30):
        alphabet = int(rng.integers(2, 6))
        eos_id = int(rng.integers(0, alphabet + 1))
        tokens = [t for t in range(alphabet + 1) if t != eos_id]
        choices = [
            [int(rng.choice(tokens)) for _ in range(int(rng.integers(1, 6)))]
            for _ in range(int(rng.integers(1, 9)))
        ]
        extra = [
            [int(rng.choice(tokens)) for _ in range(int(rng.integers(1, 6)))] for _ in range(8)
        ]
        check_everywhere(ctx, choices, eos_id, extra)


@case("two tries live side by side, and queries do not change a trie")
def independent_tries(ctx):
    first = ctx.call("build_trie", [[1, 2], [3]], 0)
    second = ctx.call("build_trie", [[1, 4], [1]], 9)
    for _ in range(2):
        check.equal(ask(ctx, first, [1]), [2], label="first trie after [1]")
        check.equal(ask(ctx, second, [1]), [4, 9], label="second trie after [1]")
        check.equal(ask(ctx, first, [1, 2, 2]), [], label="first trie off the trie")
        check.equal(ask(ctx, first, []), [1, 3], label="first trie at the start")
        check.equal(ask(ctx, second, []), [1], label="second trie at the start")
    choices = [[5, 6], [5]]
    ctx.call("build_trie", choices, 0)
    check.equal(choices, [[5, 6], [5]], label="the caller's `choices` after build_trie")
