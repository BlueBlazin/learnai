"""Tests for radix-prefix-cache.

`Model` is the statement without a tree. It keeps only the set of node prefixes (a
node is the tuple of all tokens from the root through its edge) with a lock count and
an access time each, and derives everything else by brute force: the matched length
is the longest common prefix with any stored prefix, a node's parent is its longest
proper prefix in the set, a leaf is a prefix that no other one extends. `play` runs a
script on the solution and compares, after every operation, the returned value,
`nodes()`, and `total_tokens()` with the model. `nodes()` exposes the edge boundaries
and the lock counts, so splits and locks are checked exactly; access times are checked
through the order in which `evict` removes leaves.
"""

from learnai.judge import CheckFailure, case, check


def common_prefix(a, b):
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    return n


class Model:
    def __init__(self):
        self.nodes = {}  # prefix tuple -> [locks, last access time]
        self.clock = 0

    def parent(self, prefix):
        for cut in range(len(prefix) - 1, 0, -1):
            if prefix[:cut] in self.nodes:
                return prefix[:cut]
        return ()

    def edge(self, prefix):
        return len(prefix) - len(self.parent(prefix))

    def walk(self, tokens):
        tokens = tuple(tokens)
        self.clock += 1
        length = max((common_prefix(tokens, prefix) for prefix in self.nodes), default=0)
        end = tokens[:length]
        if length and end not in self.nodes:
            # The match ends inside the edge of the shortest node that extends it.
            below = min((p for p in self.nodes if p[:length] == end), key=len)
            self.nodes[end] = list(self.nodes[below])
        for cut in range(1, length + 1):
            if tokens[:cut] in self.nodes:
                self.nodes[tokens[:cut]][1] = self.clock
        return length

    def insert(self, tokens):
        length = self.walk(tokens)
        if length < len(tokens):
            self.nodes[tuple(tokens)] = [0, self.clock]
        return length

    def lock(self, prefix, delta):
        for cut in range(1, len(prefix) + 1):
            if prefix[:cut] in self.nodes:
                self.nodes[prefix[:cut]][0] += delta

    def evict(self, n_tokens):
        evicted = 0
        while evicted < n_tokens:
            leaves = [
                p
                for p, (locks, _) in self.nodes.items()
                if locks == 0 and not any(q != p and q[: len(p)] == p for q in self.nodes)
            ]
            if not leaves:
                break
            victim = min(leaves, key=lambda p: self.nodes[p][1])
            if sum(self.nodes[p][1] == self.nodes[victim][1] for p in leaves) > 1:
                raise RuntimeError("authoring error: two candidate leaves share an access time")
            evicted += self.edge(victim)
            del self.nodes[victim]
        return evicted

    def dump(self):
        return sorted((p, self.edge(p), locks) for p, (locks, _) in self.nodes.items())

    def total(self):
        return sum(self.edge(p) for p in self.nodes)


def show(dump):
    if not dump:
        return "  (empty tree)"
    return "\n".join(f"  {list(p)}  edge {e}, locks {k}" for p, e, k in dump)


def play(ctx, ops, expect=None):
    """Run `ops` on the solution; check every step. Returns the solution's cache.

    An op is ("insert", tokens), ("match", tokens, name) which stores the returned
    handle under `name`, ("lock", name), ("unlock", name), or ("evict", n_tokens).
    `expect`, if given, lists the value each op returns according to the statement
    (the matched length for a match, None for lock and unlock).
    """
    cache = ctx.solution.RadixCache()
    model = Model()
    log = ["RadixCache()"]
    handles = {}  # name -> (the solution's handle, the prefix it stands for)

    def fail(message, extra=""):
        raise CheckFailure(message, "\n".join(log) + "\n" + extra)

    def note(index, outcome):
        if expect is not None and expect[index] != outcome:
            raise RuntimeError(f"authoring error: op {index} gives {outcome}")

    def compare():
        got = cache.nodes()
        ok = isinstance(got, list) and all(
            isinstance(row, tuple)
            and len(row) == 3
            and isinstance(row[0], tuple)
            and all(type(t) is int for t in row[0])
            and type(row[1]) is int
            and type(row[2]) is int
            for row in got
        )
        if not ok:
            fail("nodes must return a list of (prefix tuple, edge_length, locks) tuples")
        want = model.dump()
        detail = f"nodes() =\n{show(got)}\nexpected\n{show(want)}\n"
        if got != sorted(got):
            fail("nodes: the list is not sorted", detail)
        if [row[0] for row in got] != [row[0] for row in want]:
            fail(
                "nodes: the tree does not have the expected nodes (an edge was not split, a "
                "node was not removed, or a node that should remain is gone)",
                detail,
            )
        if [row[1] for row in got] != [row[1] for row in want]:
            fail("nodes: an edge length is not the number of tokens on the node's edge", detail)
        if [row[2] for row in got] != [row[2] for row in want]:
            fail("nodes: a lock count is not the number of locks that cover the node", detail)
        total = cache.total_tokens()
        if type(total) is not int or total != model.total():
            fail(
                f"total_tokens: returned {total!r}, which is not the number of tokens on the "
                "edges of the tree",
                f"expected {model.total()}\n{detail}",
            )

    for index, op in enumerate(ops):
        kind = op[0]
        if kind == "insert":
            tokens = list(op[1])
            want = model.insert(tokens)
            note(index, want)
            given = list(tokens)
            got = cache.insert(given)
            log.append(f"insert({tokens}) -> {got!r}")
            if given != tokens:
                fail("insert modified the token list it was given")
            if type(got) is not int or got != want:
                fail(
                    f"insert: returned {got!r}, which is not the length of the prefix that was "
                    "already in the tree",
                    f"expected {want}\n",
                )
        elif kind == "match":
            tokens = list(op[1])
            want = model.walk(tokens)
            note(index, want)
            got = cache.match_prefix(list(tokens))
            if not (isinstance(got, tuple) and len(got) == 2 and type(got[0]) is int):
                fail("match_prefix must return a tuple (length, node)")
            log.append(f"{op[2]} = match_prefix({tokens}) -> length {got[0]}")
            if got[0] != want:
                kind_of = "shorter" if got[0] < want else "longer"
                fail(
                    f"match_prefix: matched {got[0]} tokens, which is {kind_of} than the longest "
                    "prefix of the tokens that is in the tree",
                    f"expected {want}\n",
                )
            handles[op[2]] = (got[1], tuple(tokens[:want]))
        elif kind in ("lock", "unlock"):
            handle, prefix = handles[op[1]]
            note(index, None)
            model.lock(prefix, 1 if kind == "lock" else -1)
            log.append(f"{kind}({op[1]})   # the node ending at {list(prefix)}")
            result = cache.lock(handle) if kind == "lock" else cache.unlock(handle)
            if result is not None:
                fail(f"{kind} must return None")
        else:
            want = model.evict(op[1])
            note(index, want)
            got = cache.evict(op[1])
            log.append(f"evict({op[1]}) -> {got!r}")
            if type(got) is not int:
                fail("evict must return an int")
            compare()  # which leaves went says more than the count
            if got != want:
                fail(
                    f"evict: returned {got}, which is not the number of tokens it removed",
                    f"expected {want}\n",
                )
        compare()
    return cache


@case("the worked example: two splits, a locked path, and three evictions", public=True)
def worked_example(ctx):
    ops = [
        ("insert", [1, 2, 3, 4, 5, 6]),  # one node
        ("insert", [1, 2, 3, 7, 8]),  # splits after [1, 2, 3]
        ("match", [1, 2, 3, 4, 9], "h"),  # 4: splits [4, 5, 6] into [4] and [5, 6]
        ("lock", "h"),  # locks [1, 2, 3] and [4]
        ("insert", [9, 9, 9]),
        ("evict", 4),  # [5, 6] (time 1), then [7, 8] (time 2)
        ("evict", 100),  # only [9, 9, 9]: [4] is a leaf now, but locked
        ("unlock", "h"),
        ("evict", 1),  # [4]
        ("evict", 1),  # [1, 2, 3]: a whole leaf goes, 3 tokens
    ]
    cache = play(ctx, ops, [0, 3, 4, None, 0, 4, 3, None, 1, 3])
    check.equal(cache.nodes(), [], label="nodes() at the end")
    check.equal(cache.total_tokens(), 0, label="total_tokens() at the end")


@case("matching splits an edge: the returned node ends exactly where the match ends",
      public=True)  # fmt: skip
def match_splits(ctx):
    doc = [5, 6, 7, 8, 9, 10, 11, 12]
    ops = [
        ("insert", doc),
        ("match", doc[:3], "a"),  # ends inside the only edge: split at 3
        ("match", [*doc[:5], 99, 98], "b"),  # split at 5
        ("match", doc, "c"),  # the whole sequence, three nodes deep; no split
        ("match", [4, 5, 6], "d"),  # nothing: length 0, the root
        ("match", [], "e"),
        ("match", [*doc, 13, 14], "f"),  # 8: a longer prompt matches all of it
        ("lock", "a"),
        ("lock", "c"),
        ("lock", "d"),  # the root handle: a lock on it changes nothing
        ("unlock", "a"),
    ]
    play(ctx, ops, [0, 3, 5, 8, 0, 0, 8, None, None, None, None])


@case("insert returns the shared length and adds one leaf; a prefix of a node adds none",
      public=True)  # fmt: skip
def insert_shapes(ctx):
    ops = [
        ("insert", [1, 2, 3, 4]),
        ("insert", [1, 2, 3, 4]),  # already there: 4, nothing changes
        ("insert", [1, 2]),  # a prefix: the edge is split, no leaf is added
        ("insert", [1, 2, 3, 4, 5, 6]),  # extends a leaf: one child
        ("insert", [1, 9]),  # diverges inside [1, 2]: split after [1]
        ("insert", [7]),  # a second child of the root
        ("insert", [1, 2, 3, 8, 8]),  # diverges inside [3, 4]
        ("insert", []),
    ]
    cache = play(ctx, ops, [0, 4, 2, 4, 1, 0, 3, 0])
    check.equal(cache.total_tokens(), 10, label="total_tokens()")


@case("eviction is least recently used, leaves only, and cascades to a childless parent",
      public=True)  # fmt: skip
def lru_leaves(ctx):
    ops = [
        ("insert", [1, 1, 1, 1]),  # time 1
        ("insert", [2, 2, 2]),  # time 2
        ("insert", [1, 1, 3, 3]),  # time 3: [1, 1] -> [1, 1] and [3, 3]; [1, 1] is stamped 3
        ("insert", [4]),  # time 4
        ("match", [2, 2, 2], "x"),  # time 5: [2, 2, 2] is now the newest
        ("evict", 2),  # the lower [1, 1] (time 1), not the inner node above it
        ("evict", 3),  # [3, 3] (time 3), then its parent [1, 1] (time 3), before [4]
        ("evict", 1),  # [4] (time 4)
        ("evict", 1),  # [2, 2, 2] (time 5): all 3 tokens
        ("evict", 5),  # nothing left
    ]
    play(ctx, ops, [0, 0, 2, 0, 3, 2, 4, 1, 3, 0])


@case("a lock covers the node and every ancestor; locked leaves are skipped by evict")
def locks(ctx):
    ops = [
        ("insert", [1, 2, 3, 4, 5]),
        ("insert", [1, 2, 6]),
        ("insert", [8, 8]),
        ("match", [1, 2, 3, 4, 5], "deep"),
        ("lock", "deep"),
        ("lock", "deep"),  # two requests on the same prefix
        ("match", [1, 2, 6], "side"),
        ("lock", "side"),
        ("evict", 50),  # only [8, 8]
        ("unlock", "deep"),
        ("evict", 50),  # still locked once
        ("unlock", "side"),
        ("evict", 1),  # [6]
        ("unlock", "deep"),
        ("evict", 50),  # [3, 4, 5], then [1, 2]
    ]
    play(ctx, ops, [0, 2, 0, 5, None, None, 3, None, 2, None, 0, None, 1, None, 5])


@case("splitting a locked edge: the upper half inherits the lock count")
def split_locked_edge(ctx):
    ops = [
        ("insert", [1, 2, 3, 4, 5, 6]),
        ("match", [1, 2, 3, 4, 5, 6], "whole"),
        ("lock", "whole"),
        ("insert", [1, 2, 3, 9]),  # splits the locked edge after 3 tokens
        ("match", [1, 2], "top"),  # splits the upper half again
        ("evict", 50),  # only [9]
        ("unlock", "whole"),  # releases all three nodes of the path
        ("evict", 50),  # [4, 5, 6], [3], [1, 2]
    ]
    play(ctx, ops, [0, 6, None, 3, 2, 1, None, 6])


@case("an access stamps every node on the matched path, not only the last one")
def ancestors_touched(ctx):
    # An inner node's time only shows once it is a leaf, and it normally becomes one
    # after its newest child is gone. A lock on an unrelated leaf lets the children go
    # first while that leaf, of middle age, survives to be compared with the parent.
    ops = [
        ("insert", [1, 2, 3, 4]),  # time 1
        ("insert", [1, 2, 5, 5]),  # time 2: [1, 2] is created and stamped 2
        ("insert", [9, 9]),  # time 3
        ("match", [9, 9], "nine"),  # time 4
        ("lock", "nine"),
        ("match", [1, 2, 3, 4], "deep"),  # time 5: stamps [3, 4] and its parent [1, 2]
        ("evict", 4),  # [5, 5] and [3, 4]; [1, 2] is a leaf now, [9, 9] is locked
        ("unlock", "nine"),
        ("evict", 2),  # [9, 9] (time 4) is older than [1, 2] (time 5)
        ("evict", 2),
    ]
    play(ctx, ops, [0, 2, 0, 2, None, 4, 4, None, 2, 2])


@case("a node whose tokens were only partly matched keeps its old access time")
def partial_edge_not_touched(ctx):
    ops = [
        ("insert", [1, 2, 3, 4]),  # time 1
        ("insert", [7, 7]),  # time 2
        ("match", [1, 2, 9], "h"),  # time 3: [1, 2] is stamped 3; the lower [3, 4] keeps 1
        ("evict", 2),  # [3, 4] (time 1), not [7, 7] (time 2)
        ("insert", [5, 5, 5, 5]),  # time 4
        ("insert", [5, 5, 6]),  # time 5: the lower [5, 5] keeps time 4
        ("match", [7, 7], "s"),  # time 6
        ("evict", 2),  # [1, 2] (time 3)
        ("evict", 2),  # the lower [5, 5] (time 4), not [6] (time 5)
    ]
    play(ctx, ops, [0, 0, 2, 2, 0, 2, 2, 2, 2])


@case("evict stops as soon as n_tokens are gone, and counts whole leaves")
def evict_amounts(ctx):
    ops = [
        ("insert", [1, 1, 1]),
        ("insert", [2, 2, 2, 2]),
        ("insert", [3]),
        ("insert", [4, 4]),
        ("evict", 0),  # nothing
        ("evict", 3),  # exactly the first leaf: stop
        ("evict", 5),  # [2, 2, 2, 2] and [3]: 5
        ("evict", 1),  # [4, 4]: 2
    ]
    play(ctx, ops, [0, 0, 0, 0, 0, 3, 5, 2])


def random_ops(rng, n_ops):
    """Sequences built from a few shared stems over a tiny alphabet, so that matches
    end inside edges all the time. A handle is locked right after it is returned and
    used for nothing else once it is unlocked, so every handle the script uses is
    alive."""
    stems = [[int(t) for t in rng.integers(0, 3, size=int(rng.integers(2, 9)))] for _ in range(4)]

    def sequence():
        stem = stems[int(rng.integers(len(stems)))]
        head = stem[: int(rng.integers(0, len(stem) + 1))]
        return head + [int(t) for t in rng.integers(0, 3, size=int(rng.integers(0, 5)))]

    ops, locked, count = [], [], 0
    for _ in range(n_ops):
        roll = rng.random()
        if roll < 0.35:
            ops.append(("insert", sequence()))
        elif roll < 0.6:
            name = f"h{count}"
            count += 1
            ops.append(("match", sequence(), name))
            if rng.random() < 0.6:
                ops.append(("lock", name))
                locked.append(name)
                if rng.random() < 0.2:
                    ops.append(("lock", name))
                    locked.append(name)
        elif roll < 0.8 and locked:
            ops.append(("unlock", locked.pop(int(rng.integers(len(locked))))))
        else:
            ops.append(("evict", int(rng.integers(0, 8))))
    return ops


@case("property: random inserts, matches, locks, and evictions against the brute-force model")
def random_workloads(ctx):
    def prop(rng):
        play(ctx, random_ops(rng, 70))

    check.forall(prop, trials=40)


@case("property: long runs with frequent evictions, so nodes are split, emptied, and rebuilt")
def churn(ctx):
    def prop(rng):
        ops = random_ops(rng, 200)
        play(ctx, [op if op[0] != "evict" else ("evict", op[1] * 3) for op in ops])

    check.forall(prop, trials=6)


@case("a conversation that grows turn by turn: each turn matches everything before it")
def growing_conversation(ctx):
    # Three users share a 12-token system prompt; each has four turns of 5 tokens. A
    # turn matches the conversation so far, locks it, inserts itself, and unlocks.
    ops = []
    system = [100 + i for i in range(12)]
    for user in range(3):
        history = list(system)
        for turn in range(4):
            history = [*history, *[1000 * (user + 1) + 10 * turn + k for k in range(5)]]
            name = f"u{user}t{turn}"
            ops += [("match", history, name), ("lock", name), ("insert", history), ("unlock", name)]
    cache = play(ctx, ops)
    check.equal(cache.total_tokens(), 12 + 3 * 20, label="total_tokens()")
    check.equal(cache.evict(10**6), 12 + 3 * 20, label="evict of everything")
