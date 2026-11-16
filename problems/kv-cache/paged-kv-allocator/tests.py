"""Tests for paged-kv-allocator.

`play` runs a script of operations on the solution's allocator and, after every
operation, checks it in two ways: against invariants that follow from the statement
(no block owned twice, ids in range, free plus owned equals the pool, each sequence
owns ceil(tokens / block_size) blocks), and against `Model`, a deliberately plain
second implementation of the specification kept in this file.
"""

from learnai.judge import CheckFailure, case, check

OUT = "OutOfBlocks"  # the expected outcome of an append the pool cannot satisfy


class Model:
    """The specification, written for obviousness: a sorted scan instead of a heap."""

    def __init__(self, n_blocks, block_size):
        self.n_blocks, self.block_size = n_blocks, block_size
        self.tables, self.tokens = {}, {}

    def free_ids(self):
        owned = {block for table in self.tables.values() for block in table}
        return [block for block in range(self.n_blocks) if block not in owned]

    def append(self, seq_id, n_tokens):
        total = self.tokens.get(seq_id, 0) + n_tokens
        have = len(self.tables.get(seq_id, []))
        needed = (total + self.block_size - 1) // self.block_size - have
        free = self.free_ids()
        if needed > len(free):
            return OUT
        self.tables[seq_id] = self.tables.get(seq_id, []) + free[:needed]
        self.tokens[seq_id] = total
        return free[:needed]

    def free(self, seq_id):
        del self.tables[seq_id], self.tokens[seq_id]


def play(ctx, n_blocks, block_size, ops, expect=None):
    """Run `ops` on the solution; check every step. Returns the solution's allocator.

    An op is ("append", seq_id, n_tokens) or ("free", seq_id). `expect`, if given,
    lists the outcome of each op from the statement: the new block ids or OUT for an
    append, None for a free.
    """
    out_of_blocks = ctx.solution.OutOfBlocks
    check.that(
        isinstance(out_of_blocks, type) and issubclass(out_of_blocks, Exception),
        "OutOfBlocks must be an exception class",
    )
    allocator = ctx.solution.BlockAllocator(n_blocks, block_size)
    model = Model(n_blocks, block_size)
    log = [f"BlockAllocator(n_blocks={n_blocks}, block_size={block_size})"]
    seen = []  # every sequence id used so far, live or not

    def fail(message, extra=""):
        raise CheckFailure(message, "\n".join(log) + "\n" + extra)

    def table_of(seq_id):
        table = allocator.block_table(seq_id)
        if not isinstance(table, list) or any(type(block) is not int for block in table):
            fail("block_table must return a list of ints", f"block_table({seq_id!r}) = {table!r}\n")
        return table

    def verify():
        free = allocator.free_blocks()
        if type(free) is not int:
            fail("free_blocks must return an int", f"got {free!r}\n")
        owner, owned = {}, 0
        for seq_id in model.tables:
            try:
                table = table_of(seq_id)
            except KeyError:
                fail("block_table raised KeyError for a live sequence", f"sequence {seq_id!r}\n")
            for block in table:
                if not 0 <= block < n_blocks:
                    fail("a block table holds an id outside the pool", f"{seq_id!r}: {table}\n")
                if block in owner:
                    fail(
                        "a block is owned twice",
                        f"block {block} is in the table of {owner[block]!r} and of {seq_id!r}\n",
                    )
                owner[block] = seq_id
            owned += len(table)
            tokens = model.tokens[seq_id]
            if len(table) != -(-tokens // block_size):
                fail(
                    "a sequence does not own ceil(tokens / block_size) blocks",
                    f"{seq_id!r} holds {tokens} tokens in {len(table)} blocks: {table}\n",
                )
        if free + owned != n_blocks:
            fail(
                "free blocks plus owned blocks do not add up to the pool",
                f"free_blocks() = {free}, blocks in block tables = {owned}\n",
            )
        for seq_id, expected in model.tables.items():
            if table_of(seq_id) != expected:
                fail(
                    "a block table differs from the specification",
                    f"block_table({seq_id!r}) = {table_of(seq_id)}, expected {expected}\n",
                )
        for seq_id in seen:
            if seq_id not in model.tables:
                check.raises(KeyError, allocator.block_table, seq_id, label="block_table of a "
                             "sequence that is not live")  # fmt: skip

    for index, op in enumerate(ops):
        if op[0] == "append":
            _, seq_id, n_tokens = op
            if seq_id not in seen:
                seen.append(seq_id)
            expected = model.append(seq_id, n_tokens)
            if expect is not None and expected != expect[index]:
                raise RuntimeError(f"authoring error: op {index} gives {expected}")
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
                if not isinstance(got, list) or any(type(block) is not int for block in got):
                    fail("append must return the list of newly allocated block ids")
                if got != expected:
                    fail("append returned the wrong new blocks", f"expected {expected}\n")
        else:
            _, seq_id = op
            log.append(f"free({seq_id!r})")
            if seq_id in model.tables:
                model.free(seq_id)
                result = allocator.free(seq_id)
                if result is not None:
                    fail("free must return None")
            else:
                if expect is not None and expect[index] is not KeyError:
                    raise RuntimeError(f"authoring error: op {index} raises KeyError")
                log[-1] += " -> KeyError expected"
                try:
                    check.raises(KeyError, allocator.free, seq_id, label="free of a sequence that "
                                 "is not live")  # fmt: skip
                except CheckFailure as failure:
                    fail(failure.message)
        verify()
    return allocator


@case("the worked example: 6 blocks of 4 tokens, three sequences", public=True)
def worked_example(ctx):
    ops = [
        ("append", "a", 5),
        ("append", "b", 3),
        ("append", "a", 3),
        ("append", "a", 1),
        ("free", "b"),
        ("append", "c", 9),
        ("append", "a", 4),
        ("free", "c"),
        ("append", "a", 4),
    ]
    expect = [[0, 1], [2], [], [3], None, [2, 4, 5], OUT, None, [2]]
    allocator = play(ctx, 6, 4, ops, expect)
    check.equal(allocator.block_table("a"), [0, 1, 3, 2], label="block_table('a')")
    check.equal(allocator.free_blocks(), 2, label="free_blocks()")


@case("decoding one token at a time: a new block only every block_size tokens", public=True)
def token_by_token(ctx):
    # 7 prompt tokens, then 10 decode steps, as in Figure 6 of the PagedAttention paper.
    play(ctx, 8, 4, [("append", 0, 7), *[("append", 0, 1)] * 10])
    play(ctx, 5, 1, [("append", "s", 1)] * 5)  # block_size 1: a block per token is right


@case("OutOfBlocks changes nothing: the pool, the tables, and unknown sequences", public=True)
def refused_append(ctx):
    ops = [
        ("append", "x", 6),  # blocks 0, 1; 2 slots left in block 1
        ("append", "y", 4),  # block 2
        ("append", "x", 11),  # needs 3 blocks, 1 is free
        ("append", "new", 5),  # needs 2 blocks, 1 is free: "new" must not exist afterwards
        ("free", "new"),  # KeyError
        ("append", "x", 2),  # still fits in block 1
        ("append", "new", 4),  # the last block
        ("append", "y", 1),  # y's block is full and the pool is empty
    ]
    play(ctx, 4, 4, ops, [[0, 1], [2], OUT, OUT, KeyError, [], [3], OUT])


@case("freed blocks are reused lowest id first, whatever the order of the frees", public=True)
def lowest_first(ctx):
    ops = [
        ("append", "a", 4),  # 0, 1
        ("append", "b", 4),  # 2, 3
        ("append", "c", 4),  # 4, 5
        ("free", "c"),
        ("free", "a"),  # free: 0, 1, 4, 5 (a stack would hand out 1 or 5 first)
        ("append", "d", 5),  # 0, 1, 4
        ("free", "b"),
        ("append", "e", 3),  # 2, 3
    ]
    play(ctx, 8, 2, ops, [[0, 1], [2, 3], [4, 5], None, None, [0, 1, 4], None, [2, 3]])


@case("exact multiples of block_size: no block before the first token that needs it")
def exact_multiples(ctx):
    ops = [("append", 1, 8), ("append", 2, 4), ("append", 1, 4), ("append", 2, 1), ("append", 3, 4)]
    play(ctx, 5, 4, ops, [[0, 1], [2], [3], [4], OUT])
    play(ctx, 3, 3, [("append", "s", 9), ("free", "s"), ("append", "t", 9)], [[0, 1, 2], None] * 2)


@case("many small appends into one block, then one append spanning several blocks")
def slack_then_span(ctx):
    ops = [("append", "s", 1), ("append", "s", 2), ("append", "s", 4), ("append", "s", 20)]
    play(ctx, 6, 8, ops, [[0], [], [], [1, 2, 3]])


@case("block_table returns a copy; free and block_table raise KeyError for unknown sequences")
def copies_and_errors(ctx):
    allocator = play(ctx, 4, 2, [("append", "a", 3), ("free", "nobody"), ("append", (1, 2), 1)])
    table = allocator.block_table("a")
    table.append(99)
    table[0] = 3
    check.equal(allocator.block_table("a"), [0, 1], label="block_table('a') after editing a copy")
    check.equal(allocator.free_blocks(), 1, label="free_blocks()")
    allocator.free("a")
    check.raises(KeyError, allocator.free, "a", label="freeing a sequence twice")
    check.equal(allocator.append("z", 4), [0, 1], label="append after the free")


@case("a failed append of a new sequence does not create it, even when the pool is empty")
def no_ghost_sequence(ctx):
    ops = [("append", "big", 9), ("append", "late", 1), ("free", "late"), ("free", "big")]
    play(ctx, 3, 3, ops, [[0, 1, 2], OUT, KeyError, None])
    play(ctx, 2, 4, [("append", "huge", 9), ("append", "huge", 8)], [OUT, [0, 1]])


def random_ops(rng, n_ops, n_seqs, max_tokens):
    """A workload: mostly appends, some frees, a few frees of sequences that are not live."""
    ops, live = [], set()
    for _ in range(n_ops):
        seq_id = int(rng.integers(n_seqs))
        roll = rng.random()
        if roll < 0.72:
            ops.append(("append", seq_id, int(rng.integers(1, max_tokens + 1))))
            live.add(seq_id)  # possibly refused; the model decides
        elif roll < 0.95 and live:
            victim = sorted(live)[int(rng.integers(len(live)))]
            ops.append(("free", victim))
            live.discard(victim)
        else:
            ops.append(("free", seq_id))
            live.discard(seq_id)
    return ops


@case("property: random appends and frees never double-own or leak a block")
def random_workloads(ctx):
    def prop(rng):
        block_size = int(rng.integers(1, 9))
        n_blocks = int(rng.integers(1, 25))
        ops = random_ops(rng, 60, n_seqs=6, max_tokens=int(rng.integers(1, 3 * block_size + 2)))
        play(ctx, n_blocks, block_size, ops)

    check.forall(prop, trials=40)


@case("property: a long run on a tight pool, where most appends are refused")
def tight_pool(ctx):
    def prop(rng):
        ops = random_ops(rng, 300, n_seqs=10, max_tokens=7)
        play(ctx, 6, 4, ops)

    check.forall(prop, trials=5)
