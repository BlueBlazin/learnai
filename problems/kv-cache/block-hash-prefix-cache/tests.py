"""Tests for block-hash-prefix-cache.

`Model` is the statement written without hashing: a cached block is stored under the
tuple of all tokens up to its end, which is what a block key stands for. `play` runs a
script of operations on the solution and, after every operation, compares what the
operation returned and then probes the whole cache through its read-only queries:
`lookup` of every token sequence the script has used, `ref_count` of every prefix
that was ever cached (KeyError for the ones that are gone), and `cached_blocks`.
Since probing must change nothing, a solution whose `lookup` has side effects shows
up as a wrong eviction later.
"""

from whetstone.judge import CheckFailure, case, check


class Model:
    def __init__(self, block_size, capacity):
        self.block_size, self.capacity = block_size, capacity
        self.refs = {}  # prefix tuple -> reference count
        self.queue = []  # unreferenced prefixes, least recently released first
        self.held = {}  # request id -> prefixes held

    def prefixes(self, tokens):
        size = self.block_size
        return [tuple(tokens[:end]) for end in range(size, len(tokens) + 1, size)]

    def lookup(self, tokens):
        hits = 0
        for prefix in self.prefixes(tokens):
            if prefix not in self.refs:
                break
            hits += 1
        return hits

    def acquire(self, request_id, tokens):
        prefixes = self.prefixes(tokens)
        hits = self.lookup(tokens)
        for prefix in prefixes[:hits]:
            self.refs[prefix] += 1
            if prefix in self.queue:
                self.queue.remove(prefix)
        held = hits
        for prefix in prefixes[hits:]:
            if len(self.refs) == self.capacity:
                if not self.queue:
                    break
                del self.refs[self.queue.pop(0)]
            self.refs[prefix] = 1
            held += 1
        self.held[request_id] = prefixes[:held]
        return hits, held

    def release(self, request_id):
        for prefix in reversed(self.held.pop(request_id)):
            self.refs[prefix] -= 1
            if self.refs[prefix] == 0:
                self.queue.append(prefix)


def play(ctx, block_size, capacity, ops, expect=None):
    """Run `ops`; check every step. Returns the solution's cache.

    An op is ("acquire", request_id, tokens), ("release", request_id), or
    ("lookup", tokens). `expect`, if given, lists each outcome from the statement:
    `(hits, held)` or ValueError, None or KeyError, an int.
    """
    cache = ctx.solution.PrefixBlockCache(block_size, capacity)
    model = Model(block_size, capacity)
    log = [f"PrefixBlockCache(block_size={block_size}, capacity={capacity})"]
    sequences, ever_cached = [], set()

    def fail(message, extra=""):
        raise CheckFailure(message, "\n".join(log) + "\n" + extra)

    def note(index, outcome):
        if expect is not None and expect[index] != outcome:
            raise RuntimeError(f"authoring error: op {index} gives {outcome}")

    def remember(tokens):
        if list(tokens) not in sequences:
            sequences.append(list(tokens))

    def probe():
        size = cache.cached_blocks()
        if type(size) is not int or size != len(model.refs):
            fail(
                f"cached_blocks: the cache holds {size!r} blocks, which is not the expected number",
                f"expected {len(model.refs)} (capacity {capacity})\n",
            )
        for tokens in reversed(sequences):  # an order a refreshing lookup would disturb
            got = cache.lookup(list(tokens))
            want = model.lookup(tokens)
            if type(got) is not int or got != want:
                kind = "a block that is not cached is found" if got > want else (
                    "a block that should be cached is not found")  # fmt: skip
                fail(
                    f"lookup: {kind} (probing the cache after the last operation)",
                    f"lookup({tokens}) = {got!r}, expected {want}\n",
                )
        ever_cached.update(model.refs)
        for prefix in sorted(ever_cached):
            if prefix in model.refs:
                try:
                    got = cache.ref_count(list(prefix))
                except KeyError:
                    fail("ref_count raised KeyError for a cached block", f"prefix {prefix}\n")
                if type(got) is not int or got != model.refs[prefix]:
                    fail(
                        "ref_count: a reference count is not the number of active requests "
                        "holding the block",
                        f"ref_count({list(prefix)}) = {got!r}, expected {model.refs[prefix]}\n",
                    )
            else:
                try:
                    check.raises(KeyError, cache.ref_count, list(prefix), label="ref_count of a "
                                 "block that is not cached")  # fmt: skip
                except CheckFailure as failure:
                    fail(failure.message, f"prefix {prefix}\n")

    for index, op in enumerate(ops):
        if op[0] == "acquire":
            _, request_id, tokens = op
            remember(tokens)
            log.append(f"acquire({request_id!r}, {list(tokens)})")
            if request_id in model.held:
                note(index, ValueError)
                log[-1] += " -> ValueError expected"
                try:
                    check.raises(ValueError, cache.acquire, request_id, list(tokens),
                                 label="acquire with an active request id")  # fmt: skip
                except CheckFailure as failure:
                    fail(failure.message)
            else:
                want = model.acquire(request_id, tokens)
                note(index, want)
                given = list(tokens)
                got = cache.acquire(request_id, given)
                log[-1] += f" -> {got!r}"
                if given != list(tokens):
                    fail("acquire modified the token list it was given")
                if not (
                    isinstance(got, tuple) and len(got) == 2 and all(type(x) is int for x in got)
                ):
                    fail("acquire must return a tuple (hits, held) of two ints")
                if got[0] != want[0]:
                    fail(
                        f"acquire: reported {got[0]} reused blocks, which is not the number of "
                        "leading blocks that were cached",
                        f"expected {want!r}\n",
                    )
                if got[1] != want[1]:
                    fail(
                        f"acquire: the request holds {got[1]} blocks, which is not the expected "
                        "number",
                        f"expected {want!r}\n",
                    )
        elif op[0] == "release":
            _, request_id = op
            log.append(f"release({request_id!r})")
            if request_id in model.held:
                note(index, None)
                model.release(request_id)
                if cache.release(request_id) is not None:
                    fail("release must return None")
            else:
                note(index, KeyError)
                log[-1] += " -> KeyError expected"
                try:
                    check.raises(KeyError, cache.release, request_id, label="release of a "
                                 "request that is not active")  # fmt: skip
                except CheckFailure as failure:
                    fail(failure.message)
        else:
            _, tokens = op
            remember(tokens)
            want = model.lookup(tokens)
            note(index, want)
            got = cache.lookup(list(tokens))
            log.append(f"lookup({list(tokens)}) -> {got!r}")
            if type(got) is not int or got != want:
                fail(
                    f"lookup: returned {got!r}, which is not the number of leading full blocks "
                    "that are cached",
                    f"expected {want}\n",
                )
        probe()
    return cache


# The worked example: block_size 4. S is a shared system prompt of two blocks.
S = [1, 2, 3, 4, 5, 6, 7, 8]
Q1 = [*S, 11, 12, 13, 14, 15, 16]  # three full blocks and two tokens
Q2 = [*S, 21, 22, 23, 24, 25]  # three full blocks and one token
T = [31, 32, 33, 34, 35, 36, 37, 38, 39]  # two full blocks and one token
X = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]  # three full blocks
Y = [61, 62, 63, 64, 65, 66, 67, 68]


@case("the worked example: a shared system prompt, LRU eviction, and a full cache",
      public=True)  # fmt: skip
def worked_example(ctx):
    ops = [
        ("acquire", "r1", Q1),  # three blocks inserted
        ("lookup", [*S, 99]),  # the system prompt's two blocks
        ("lookup", [5, 6, 7, 8, 1, 2, 3, 4]),  # the same two blocks of tokens, swapped: 0
        ("acquire", "r2", Q2),  # two hits, one insert
        ("release", "r1"),  # only Q1's third block becomes evictable
        ("acquire", "r3", T),  # the second block of T evicts it
        ("lookup", Q1),  # 2: the system prompt is still there
        ("release", "r2"),  # evictable now: Q2's third block, then S's second, then S's first
        ("acquire", "r4", X),  # evicts all three, in that order
        ("lookup", Q2),
        ("acquire", "r5", Y),  # the cache is full of referenced blocks: nothing is cached
        ("release", "r5"),
        ("release", "r3"),
        ("acquire", "r6", Y),  # T's two blocks are evicted, second block first
    ]
    expect = [(0, 3), 2, 0, (2, 3), None, (0, 2), 2, None, (0, 3), 0, (0, 0), None, None, (0, 2)]
    cache = play(ctx, 4, 5, ops, expect)
    check.equal(cache.cached_blocks(), 5, label="cached_blocks()")


@case("a block is its whole prefix: the same tokens after a different start are a miss",
      public=True)  # fmt: skip
def same_block_different_prefix(ctx):
    a, b, c, d = [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]
    ops = [
        ("acquire", "x", [*a, *b]),
        ("acquire", "y", [*c, *d]),
        ("lookup", [*a, *d]),  # 1: d is cached, but after c, not after a
        ("lookup", [*c, *b]),  # 1
        ("lookup", b),  # 0: b is cached as a second block, not as a first
        ("lookup", [*a, *b, *a, *b]),  # 2: the third block repeats a, after a b
        ("acquire", "z", [*a, *d, *b]),  # 1 hit; d-after-a and b-after-a-d are new blocks
        ("lookup", [*a, *d, *b]),
    ]
    play(ctx, 3, 16, ops, [(0, 2), (0, 2), 1, 1, 0, 2, (1, 3), 3])


@case("only full blocks are cached: a partial last block is never a hit", public=True)
def partial_block(ctx):
    ops = [
        ("acquire", "p", [1, 2, 3, 4, 5, 6]),  # one full block; 5, 6 are not cached
        ("lookup", [1, 2, 3, 4, 5, 6]),  # 1, not 2
        ("lookup", [1, 2, 3, 4, 5, 6, 7, 8]),  # 1: the block 5 6 7 8 was never computed
        ("lookup", [1, 2, 3]),  # 0: no full block
        ("acquire", "q", [1, 2]),  # holds nothing
        ("release", "q"),
        ("acquire", "r", [1, 2, 3, 4, 5, 6, 7, 8]),
        ("lookup", [1, 2, 3, 4, 5, 6]),
    ]
    cache = play(ctx, 4, 4, ops, [(0, 1), 1, 1, 0, (0, 0), None, (1, 2), 1])
    check.raises(KeyError, cache.ref_count, [1, 2, 3], label="ref_count of tokens without a "
                 "full block")  # fmt: skip
    check.equal(cache.ref_count([1, 2, 3, 4, 5]), 2, label="ref_count([1, 2, 3, 4, 5])")


@case("a block held by two requests stays until both release it", public=True)
def two_holders(ctx):
    a, b, c = [1, 2], [3, 4], [5, 6]
    ops = [
        ("acquire", 1, [*a, *b]),
        ("acquire", 2, [*a, *b]),  # two hits
        ("release", 1),  # both blocks are still referenced by request 2
        ("acquire", 3, [*c, *c, *c]),  # capacity 3: one free slot, nothing evictable
        ("release", 2),
        ("acquire", 4, [*b, *a]),  # evicts a-b's second block, then its first
        ("lookup", [*a, *b]),
    ]
    play(ctx, 2, 3, ops, [(0, 2), (2, 2), None, (0, 1), None, (0, 2), 0])


@case("least recently released first: a reused block moves to the back of the order")
def lru_order(ctx):
    a, b, c, d, e = [1], [2], [3], [4], [5]
    ops = [
        ("acquire", "a", a),
        ("acquire", "b", b),
        ("acquire", "c", c),
        ("release", "a"),
        ("release", "b"),
        ("release", "c"),  # order: a, b, c
        ("acquire", "a2", a),  # a hit: a leaves the order
        ("release", "a2"),  # order: b, c, a
        ("acquire", "d", d),  # evicts b, not a
        ("lookup", a),
        ("lookup", b),
        ("acquire", "e", e),  # evicts c
        ("lookup", a),
    ]
    expect = [(0, 1), (0, 1), (0, 1), None, None, None, (1, 1), None, (0, 1), 1, 0, (0, 1), 1]
    play(ctx, 1, 3, ops, expect)


@case("a release queues the deepest block first, so a chain is evicted from its tail")
def tail_first(ctx):
    chain = [1, 2, 3, 4, 5, 6]  # three blocks of two
    ops = [
        ("acquire", "c", chain),
        ("release", "c"),  # order: block 3, block 2, block 1
        ("acquire", "n", [9, 9]),  # evicts the chain's third block
        ("lookup", chain),  # 2: the head of the chain is intact
        ("acquire", "m", [8, 8]),  # evicts the second
        ("lookup", chain),  # 1
        ("acquire", "again", chain),  # 1 hit; blocks 2 and 3 cannot be inserted
    ]
    play(ctx, 2, 3, ops, [(0, 3), None, (0, 1), 2, (0, 1), 1, (1, 1)])


@case("hits are referenced before room is made: a request never evicts what it reuses")
def pin_before_insert(ctx):
    doc = [1, 2, 3, 4]  # two blocks
    ops = [
        ("acquire", "old", [7, 7]),
        ("acquire", "d", doc),
        ("release", "d"),  # order: doc block 2, doc block 1
        ("release", "old"),  # order: doc 2, doc 1, old
        ("acquire", "ext", [*doc, 5, 6, 8, 8]),  # 2 hits; the 2 inserts evict old, then stop
        ("lookup", [*doc, 5, 6, 8, 8]),
        ("lookup", [7, 7]),
    ]
    play(ctx, 2, 3, ops, [(0, 1), (0, 2), None, None, (2, 3), 3, 0])


@case("a lookup changes nothing: neither the eviction order nor a reference count")
def lookup_is_read_only(ctx):
    a, b, c = [1, 1], [2, 2], [3, 3]
    ops = [
        ("acquire", 1, a),
        ("acquire", 2, b),
        ("release", 1),
        ("release", 2),  # order: a, b
        ("lookup", a),  # must not refresh a
        ("lookup", a),
        ("acquire", 3, c),  # evicts a
        ("lookup", a),
        ("lookup", b),
    ]
    play(ctx, 2, 2, ops, [(0, 1), (0, 1), None, None, 1, 1, (0, 1), 0, 1])


@case("errors: an active request id, an unknown request, and ids of any hashable type")
def errors(ctx):
    ops = [
        ("release", "nobody"),
        ("acquire", ("req", 1), [1, 2, 3, 4]),
        ("acquire", ("req", 1), [5, 6, 7, 8]),  # ValueError, and nothing is inserted
        ("lookup", [5, 6, 7, 8]),
        ("release", ("req", 1)),
        ("release", ("req", 1)),  # KeyError: released already
        ("acquire", ("req", 1), [5, 6, 7, 8]),  # the id is free again
        ("acquire", 0, []),  # an empty prompt holds nothing
        ("release", 0),
    ]
    expect = [KeyError, (0, 2), ValueError, 0, None, KeyError, (0, 2), (0, 0), None]
    play(ctx, 2, 8, ops, expect)


def random_ops(rng, n_ops, block_size):
    """Prompts built from a few block-sized pieces, so that prompts share prefixes and
    the same piece turns up at different depths; partial tails are common."""
    pieces = [[int(t) for t in rng.integers(0, 3, size=block_size)] for _ in range(4)]

    def prompt():
        tokens = []
        for _ in range(int(rng.integers(0, 5))):
            tokens += pieces[int(rng.integers(len(pieces)))]
        return tokens + [int(t) for t in rng.integers(0, 3, size=int(rng.integers(0, block_size)))]

    ops, active, next_id = [], [], 0
    for _ in range(n_ops):
        roll = rng.random()
        if roll < 0.45 or not active:
            ops.append(("acquire", next_id, prompt()))
            active.append(next_id)
            next_id += 1
        elif roll < 0.8:
            ops.append(("release", active.pop(int(rng.integers(len(active))))))
        elif roll < 0.95:
            ops.append(("lookup", prompt()))
        elif roll < 0.975:
            ops.append(("acquire", active[int(rng.integers(len(active)))], prompt()))
        else:
            ops.append(("release", next_id + 100))
    return ops


@case("property: random prompts over shared pieces; the cache equals the model after each step")
def random_workloads(ctx):
    def prop(rng):
        block_size = int(rng.integers(1, 5))
        capacity = int(rng.integers(1, 12))
        play(ctx, block_size, capacity, random_ops(rng, 60, block_size))

    check.forall(prop, trials=40)


@case("property: a small cache under many concurrent requests, often full of referenced blocks")
def tight_cache(ctx):
    def prop(rng):
        play(ctx, 2, 4, random_ops(rng, 150, 2))

    check.forall(prop, trials=8)


@case("a long prompt: 2,000 blocks are keyed, inserted, and found again quickly",
      time_limit_s=5)  # fmt: skip
def long_prompt(ctx):
    tokens = [int(t) for t in ctx.rng.integers(0, 50_000, size=32_000)]
    cache = ctx.solution.PrefixBlockCache(16, 4096)
    check.equal(cache.acquire("long", list(tokens)), (0, 2000), label="acquire")
    check.equal(cache.lookup(tokens[:16_005]), 1000, label="lookup of half the prompt")
    changed = list(tokens)
    changed[16 * 700 + 3] += 1
    check.equal(cache.lookup(changed), 700, label="lookup with one token changed in block 700")
    check.equal(cache.acquire("again", changed), (700, 2000), label="acquire")
    check.equal(cache.cached_blocks(), 3300, label="cached_blocks()")
    check.equal(cache.ref_count(tokens[: 16 * 700]), 2, label="ref_count of a shared block")
