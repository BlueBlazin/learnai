"""Tests for paged-kv-copy-on-write.

`play` runs a script of operations on the solution's allocator and checks every step
three ways:

- against `Model`, a deliberately plain second implementation of the statement (it
  recounts owners from the block tables instead of keeping reference counts);
- against invariants: a reference count equals the number of tables holding the
  block, and free blocks plus distinct held blocks make up the pool;
- by playing the engine. The test keeps the contents of every physical block,
  performs the copies `append` returns, writes each new token into the slot the
  sequence's block table names, and then reads every live sequence back through its
  table. A sequence must read exactly the tokens it holds (its ancestors' up to the
  fork, then its own), whatever its siblings did.
"""

from whetstone.judge import CheckFailure, case, check

OUT = "OutOfBlocks"  # the expected outcome of an append the pool cannot satisfy


class Model:
    """The specification, written for obviousness."""

    def __init__(self, n_blocks, block_size):
        self.n_blocks, self.block_size = n_blocks, block_size
        self.tables, self.tokens = {}, {}

    def owners(self, block):
        return sum(block in table for table in self.tables.values())

    def free_ids(self):
        return [block for block in range(self.n_blocks) if self.owners(block) == 0]

    def append(self, seq_id, n_tokens):
        table = list(self.tables.get(seq_id, []))
        tokens = self.tokens.get(seq_id, 0)
        copy = tokens % self.block_size != 0 and self.owners(table[-1]) > 1
        growth = -(-(tokens + n_tokens) // self.block_size) - len(table)
        free = self.free_ids()
        if copy + growth > len(free):
            return OUT
        new, copies = free[: copy + growth], []
        if copy:
            copies.append((table[-1], new[0]))
            table[-1] = new[0]
            table += new[1:]
        else:
            table += new
        self.tables[seq_id], self.tokens[seq_id] = table, tokens + n_tokens
        return new, copies

    def fork(self, parent_id, child_id):
        self.tables[child_id] = list(self.tables[parent_id])
        self.tokens[child_id] = self.tokens[parent_id]

    def free(self, seq_id):
        del self.tables[seq_id], self.tokens[seq_id]


def play(ctx, n_blocks, block_size, ops, expect=None):
    """Run `ops` on the solution; check every step. Returns (allocator, totals).

    An op is ("append", seq_id, n_tokens), ("fork", parent_id, child_id), or
    ("free", seq_id). `expect`, if given, lists the outcome of each op from the
    statement: `(allocated, copies)` or OUT for an append, None for a fork or free
    that succeeds, KeyError or ValueError for one that must raise.
    """
    out_of_blocks = ctx.solution.OutOfBlocks
    check.that(
        isinstance(out_of_blocks, type) and issubclass(out_of_blocks, Exception),
        "OutOfBlocks must be an exception class",
    )
    allocator = ctx.solution.CowAllocator(n_blocks, block_size)
    model = Model(n_blocks, block_size)
    log = [f"CowAllocator(n_blocks={n_blocks}, block_size={block_size})"]
    seen = []  # every sequence id used so far, live or not
    store = {block: [None] * block_size for block in range(n_blocks)}  # the "GPU memory"
    held = {}  # seq_id -> the token values the sequence holds, in order
    totals = {"copies": 0, "allocated": 0}

    def fail(message, extra=""):
        raise CheckFailure(message, "\n".join(log) + "\n" + extra)

    def table_of(seq_id):
        try:
            table = allocator.block_table(seq_id)
        except KeyError:
            fail("block_table raised KeyError for a live sequence", f"sequence {seq_id!r}\n")
        if not isinstance(table, list) or any(type(block) is not int for block in table):
            fail("block_table must return a list of ints", f"block_table({seq_id!r}) = {table!r}\n")
        if any(not 0 <= block < n_blocks for block in table):
            fail("a block table holds an id outside the pool", f"{seq_id!r}: {table}\n")
        return table

    def verify():
        tables = {seq_id: table_of(seq_id) for seq_id in model.tables}
        # 1. Every sequence reads back its own tokens through its block table.
        for seq_id, table in tables.items():
            values = held[seq_id]
            if len(table) != -(-len(values) // block_size):
                fail(
                    "a sequence does not hold ceil(tokens / block_size) blocks",
                    f"{seq_id!r} holds {len(values)} tokens in {len(table)} blocks: {table}\n",
                )
            read = [store[table[i // block_size]][i % block_size] for i in range(len(values))]
            if read != values:
                where = next(i for i in range(len(values)) if read[i] != values[i])
                fail(
                    "a sequence reads a token it did not write: its block table points at a "
                    "block that another sequence has written to, or at a copy that was never "
                    "made",
                    f"sequence {seq_id!r}, token {where} (block {table[where // block_size]}, "
                    f"slot {where % block_size}): holds {values[where]!r}, reads {read[where]!r}\n",
                )
        # 2. Reference counts are owner counts, and nothing leaks.
        for block in range(n_blocks):
            count = allocator.ref_count(block)
            owners = sum(block in table for table in tables.values())
            if type(count) is not int or count != owners:
                fail(
                    "a reference count is not the number of sequences that hold the block",
                    f"ref_count({block}) = {count!r}; block tables holding it: {owners}\n",
                )
        free = allocator.free_blocks()
        distinct = len({block for table in tables.values() for block in table})
        if type(free) is not int or free + distinct != n_blocks:
            fail(
                "free blocks plus held blocks do not add up to the pool",
                f"free_blocks() = {free!r}, distinct blocks in block tables = {distinct}\n",
            )
        # 3. The exact tables of the specification.
        for seq_id, expected in model.tables.items():
            if tables[seq_id] != expected:
                fail(
                    "a block table differs from the specification",
                    f"block_table({seq_id!r}) = {tables[seq_id]}, expected {expected}\n",
                )
        for seq_id in seen:
            if seq_id not in model.tables:
                check.raises(KeyError, allocator.block_table, seq_id, label="block_table of a "
                             "sequence that is not live")  # fmt: skip

    def raises(exc_type, fn, *args, what):
        try:
            check.raises(exc_type, fn, *args, label=what)
        except CheckFailure as failure:
            fail(failure.message)

    def note(index, outcome):
        if expect is not None and expect[index] != outcome:
            raise RuntimeError(f"authoring error: op {index} gives {outcome}")

    for index, op in enumerate(ops):
        for seq_id in op[1:3] if op[0] == "fork" else op[1:2]:
            if seq_id not in seen:
                seen.append(seq_id)
        if op[0] == "append":
            _, seq_id, n_tokens = op
            before = model.tokens.get(seq_id, 0)
            expected = model.append(seq_id, n_tokens)
            note(index, expected)
            log.append(f"append({seq_id!r}, {n_tokens})")
            try:
                got = allocator.append(seq_id, n_tokens)
            except out_of_blocks:
                got = OUT
            log[-1] += f" -> {got!r}"
            if expected == OUT and got != OUT:
                fail("append must raise OutOfBlocks when the pool has too few free blocks")
            if got == OUT and expected != OUT:
                fail("append raised OutOfBlocks although enough blocks are free")
            if got != OUT:
                ok = (
                    isinstance(got, tuple)
                    and len(got) == 2
                    and isinstance(got[0], list)
                    and all(type(block) is int for block in got[0])
                    and isinstance(got[1], list)
                    and all(
                        isinstance(pair, tuple)
                        and len(pair) == 2
                        and all(type(b) is int and 0 <= b < n_blocks for b in pair)
                        for pair in got[1]
                    )
                )
                if not ok:
                    fail("append must return (allocated, copies): a list of ints and a list of "
                         "(src, dst) tuples of block ids")  # fmt: skip
                totals["copies"] += len(got[1])
                totals["allocated"] += len(got[0])
                # Play the engine: copy first, then write the new tokens.
                for source, target in got[1]:
                    store[target] = list(store[source])
                table = table_of(seq_id)
                values = held.setdefault(seq_id, [])
                for k in range(n_tokens):
                    position = before + k
                    value = f"{seq_id}#{position}@{index}"
                    values.append(value)
                    if position // block_size < len(table):
                        store[table[position // block_size]][position % block_size] = value
                verify()  # report a broken sequence before a mismatch of the returned ids
                if got != expected:
                    fail(
                        "append returned the wrong (allocated, copies)",
                        f"expected {expected!r}\n",
                    )
        elif op[0] == "fork":
            _, parent_id, child_id = op
            log.append(f"fork({parent_id!r}, {child_id!r})")
            if child_id in model.tables:
                note(index, ValueError)
                log[-1] += " -> ValueError expected"
                raises(ValueError, allocator.fork, parent_id, child_id, what="fork onto a live "
                       "sequence id")  # fmt: skip
            elif parent_id not in model.tables:
                note(index, KeyError)
                log[-1] += " -> KeyError expected"
                raises(KeyError, allocator.fork, parent_id, child_id, what="fork of a sequence "
                       "that is not live")  # fmt: skip
            else:
                note(index, None)
                model.fork(parent_id, child_id)
                held[child_id] = list(held[parent_id])
                try:
                    result = allocator.fork(parent_id, child_id)
                except out_of_blocks:
                    fail("fork raised OutOfBlocks: a fork shares blocks and allocates nothing")
                if result is not None:
                    fail("fork must return None")
        else:
            _, seq_id = op
            log.append(f"free({seq_id!r})")
            if seq_id in model.tables:
                note(index, None)
                model.free(seq_id)
                del held[seq_id]
                if allocator.free(seq_id) is not None:
                    fail("free must return None")
            else:
                note(index, KeyError)
                log[-1] += " -> KeyError expected"
                raises(KeyError, allocator.free, seq_id, what="free of a sequence that is not "
                       "live")  # fmt: skip
        verify()
    return allocator, totals


@case("the worked example: a fork, one copy on write, and frees in either order", public=True)
def worked_example(ctx):
    ops = [
        ("append", "a", 6),  # blocks 0, 1; block 1 holds 2 of 4
        ("fork", "a", "b"),  # b shares 0 and 1
        ("append", "b", 1),  # block 1 is shared and partly filled: b copies it to 2
        ("append", "a", 1),  # a is now the only holder of block 1: no copy
        ("append", "b", 4),  # fills block 2, then block 3
        ("fork", "b", "c"),  # c shares 0, 2, 3
        ("free", "b"),  # nothing returns to the pool: c holds all of b's blocks
        ("append", "c", 2),  # block 3 (3 of 4) is c's alone again: no copy; one new block
        ("free", "a"),  # block 1 returns; block 0 is still c's
        ("free", "c"),
    ]
    expect = [
        ([0, 1], []),
        None,
        ([2], [(1, 2)]),
        ([], []),
        ([3], []),
        None,
        None,
        ([4], []),
        None,
        None,
    ]
    allocator, _ = play(ctx, 6, 4, ops, expect)
    check.equal(allocator.free_blocks(), 6, label="free_blocks() at the end")


@case("parallel sampling: four sequences share a 10-token prompt, 3 copies, 14 blocks",
      public=True)  # fmt: skip
def parallel_sampling(ctx):
    # The prompt fills blocks 0 and 1 and half of block 2. Each of the first three
    # sequences to decode copies block 2; the fourth is then its only holder.
    seqs = ["s0", "s1", "s2", "s3"]
    ops = [("append", "s0", 10)] + [("fork", "s0", seq) for seq in seqs[1:]]
    ops += [("append", seq, 1) for _ in range(7) for seq in seqs]
    allocator, totals = play(ctx, 16, 4, ops)
    check.equal(allocator.ref_count(0), 4, label="ref_count of the first prompt block")
    check.at_most("block_copies", totals["copies"], 3, unit="copies")
    check.at_most("blocks_allocated", totals["allocated"], 14, unit="blocks")


@case("a fork at a block boundary never copies: a full block is not written again",
      public=True)  # fmt: skip
def full_last_block(ctx):
    ops = [
        ("append", "p", 8),  # blocks 0, 1, both full
        ("fork", "p", "q"),
        ("append", "q", 3),  # a new block, no copy
        ("append", "p", 5),  # two new blocks, no copy
        ("fork", "q", "r"),  # r shares q's partly filled block 2
        ("append", "r", 6),  # copies block 2, then one more block
    ]
    expect = [([0, 1], []), None, ([2], []), ([3, 4], []), None, ([5, 6, 7], [(2, 5)])]
    play(ctx, 8, 4, ops, expect)


@case("OutOfBlocks counts the copy, and a refused append changes nothing", public=True)
def refused_append(ctx):
    ops = [
        ("append", "a", 5),  # blocks 0, 1
        ("fork", "a", "b"),
        ("append", "b", 2),  # copy of block 1 into 2
        ("fork", "a", "c"),  # c shares block 1 with a
        ("append", "c", 4),  # needs the copy and one more block; only block 3 is free
        ("append", "c", 3),  # needs only the copy: block 3
        ("append", "a", 1),  # a holds block 1 alone again; fits
        ("append", "a", 3),  # block 1 is full now; the pool is empty
        ("free", "b"),  # block 2 returns (block 0 is still shared)
        ("append", "a", 3),
    ]
    expect = [
        ([0, 1], []),
        None,
        ([2], [(1, 2)]),
        None,
        OUT,
        ([3], [(1, 3)]),
        ([], []),
        OUT,
        None,
        ([2], []),
    ]
    play(ctx, 4, 4, ops, expect)


@case("the parent finishes first: its blocks stay with the children until the last one frees")
def parent_freed_first(ctx):
    ops = [
        ("append", "root", 7),  # 0, 1
        ("fork", "root", "x"),
        ("fork", "root", "y"),
        ("free", "root"),
        ("append", "new", 4),  # must get block 2, not a block x and y still read
        ("append", "x", 1),  # copy of 1 into 3
        ("append", "y", 1),  # y is the last holder of block 1: no copy
        ("free", "x"),  # block 3 returns, block 0 does not
        ("append", "new", 1),  # block 3
        ("free", "y"),  # blocks 0 and 1 return
        ("append", "new", 9),  # 14 tokens: two more blocks, 0 and 1
    ]
    expect = [
        ([0, 1], []),
        None,
        None,
        None,
        ([2], []),
        ([3], [(1, 3)]),
        ([], []),
        None,
        ([3], []),
        None,
        ([0, 1], []),
    ]
    play(ctx, 6, 4, ops, expect)


@case("the last holder of a block writes in place, also after several forks")
def last_holder_writes_in_place(ctx):
    ops = [
        ("append", 0, 3),  # block 0, 3 of 8
        ("fork", 0, 1),
        ("fork", 1, 2),
        ("append", 0, 1),  # copy to 1
        ("append", 1, 1),  # copy to 2
        ("append", 2, 1),  # in place
        ("fork", 2, 3),
        ("free", 3),  # the fork is gone again: 2 holds block 0 alone
        ("append", 2, 1),  # in place
    ]
    expect = [
        ([0], []),
        None,
        None,
        ([1], [(0, 1)]),
        ([2], [(0, 2)]),
        ([], []),
        None,
        None,
        ([], []),
    ]
    play(ctx, 4, 8, ops, expect)


@case("the copy destination is the lowest free id, ahead of the blocks for new tokens")
def allocation_order(ctx):
    ops = [
        ("append", "a", 2),  # 0
        ("append", "b", 8),  # 1, 2
        ("append", "a", 1),  # in place
        ("free", "b"),
        ("fork", "a", "c"),
        ("append", "c", 9),  # copy into 1, then 2 and 3
        ("append", "a", 6),  # a holds block 0 alone: no copy, blocks 4 and 5
    ]
    expect = [([0], []), ([1, 2], []), ([], []), None, None, ([1, 2, 3], [(0, 1)]), ([4, 5], [])]
    play(ctx, 6, 4, ops, expect)


@case("fork errors: an unknown parent, a live child id, a fork onto itself")
def fork_errors(ctx):
    ops = [
        ("fork", "ghost", "a"),  # KeyError, and "a" must not exist afterwards
        ("free", "a"),
        ("append", "a", 3),
        ("append", "b", 3),
        ("fork", "a", "b"),  # ValueError: b is live
        ("fork", "a", "a"),  # ValueError
        ("fork", "ghost", "b"),  # ValueError: the child is checked first
        ("fork", "a", (1, "tuple id")),
        ("append", (1, "tuple id"), 2),
        ("free", "a"),
        ("fork", "a", "z"),  # KeyError: a is gone
    ]
    expect = [
        KeyError,
        KeyError,
        ([0], []),
        ([1], []),
        ValueError,
        ValueError,
        ValueError,
        None,
        ([2, 3], [(0, 2)]),
        None,
        KeyError,
    ]
    allocator, _ = play(ctx, 5, 4, ops, expect)
    table = allocator.block_table((1, "tuple id"))
    table.append(99)
    check.equal(allocator.block_table((1, "tuple id")), [2, 3], label="block_table after "
                "editing the returned list")  # fmt: skip


@case("a refused append of a forked sequence leaves tables, counts, and the pool untouched")
def refused_forked_append(ctx):
    ops = [
        ("append", "a", 6),  # 0, 1 (bs 4)
        ("fork", "a", "b"),
        ("append", "c", 4),  # 2: the pool is now empty
        ("append", "b", 1),  # the copy alone does not fit
        ("append", "a", 1),  # neither for a: block 1 is still shared
        ("append", "b", 7),
        ("free", "c"),
        ("append", "b", 7),  # copy into 2, but 2 more blocks are needed: still refused
        ("append", "b", 2),  # copy into 2 only
        ("append", "a", 2),  # a is alone on block 1
    ]
    expect = [([0, 1], []), None, ([2], []), OUT, OUT, OUT, None, OUT, ([2], [(1, 2)]), ([], [])]
    play(ctx, 3, 4, ops, expect)


def random_ops(rng, n_ops, n_seqs, max_tokens):
    """Appends, forks, and frees, with a few invalid forks and frees mixed in."""
    ops, live = [], set()
    for _ in range(n_ops):
        seq_id = int(rng.integers(n_seqs))
        roll = rng.random()
        if roll < 0.5 or not live:
            ops.append(("append", seq_id, int(rng.integers(1, max_tokens + 1))))
            live.add(seq_id)  # possibly refused; the model decides
        elif roll < 0.75:
            parent = sorted(live)[int(rng.integers(len(live)))]
            ops.append(("fork", parent, seq_id))
            live.add(seq_id)
        elif roll < 0.95:
            victim = sorted(live)[int(rng.integers(len(live)))]
            ops.append(("free", victim))
            live.discard(victim)
        else:
            ops.append(("fork", seq_id, int(rng.integers(n_seqs))))
    return ops


@case("property: random appends, forks, and frees; every sequence reads its own tokens")
def random_workloads(ctx):
    def prop(rng):
        block_size = int(rng.integers(1, 7))
        n_blocks = int(rng.integers(2, 30))
        ops = random_ops(rng, 80, n_seqs=7, max_tokens=int(rng.integers(1, 2 * block_size + 2)))
        play(ctx, n_blocks, block_size, ops)

    check.forall(prop, trials=40)


@case("property: a tight pool of 6 blocks, where many appends are refused for want of a copy")
def tight_pool(ctx):
    def prop(rng):
        play(ctx, 6, 4, random_ops(rng, 250, n_seqs=6, max_tokens=5))

    check.forall(prop, trials=6)


@case("beam search: each step forks the survivors and frees the losers; blocks never leak")
def beam_search(ctx):
    # Width 3, block size 4, 12 steps. At every step each beam forks one child, both
    # append a token, and the three oldest sequences are freed.
    ops = [("append", "b0", 5), ("fork", "b0", "b1"), ("fork", "b0", "b2")]
    beams, counter = ["b0", "b1", "b2"], 3
    for _ in range(12):
        children = []
        for beam in beams:
            child = f"b{counter}"
            counter += 1
            ops += [("fork", beam, child), ("append", beam, 1), ("append", child, 1)]
            children.append(child)
        ops += [("free", beam) for beam in beams]
        beams = children
    ops += [("free", beam) for beam in beams]
    allocator, totals = play(ctx, 40, 4, ops)
    check.equal(allocator.free_blocks(), 40, label="free_blocks() after every beam is freed")
    ctx.metric("block_copies", totals["copies"], "copies")
