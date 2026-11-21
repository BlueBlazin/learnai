"""Tests for kv-tiered-offload.

`SimTiers` is the store the statement describes: three tiers of block slots on a
simulated clock. It executes what the solution tells it, takes virtual time for
loads and computations, counts the bytes that cross a tier boundary, and refuses
(with a `TierError`, also recorded in `store.violations`) anything that cannot be
done: a full tier, a block that is not where the call says, a block loaded or
computed while its parent is not on the GPU, two operations at once. It learns what
a key stands for from `compute(key, parent, tokens)`: the block's prefix is the
parent's prefix plus these tokens. So every operation is logged under the prefix,
not under the solution's key, the logs of two implementations can be compared, and a
key that stands for two prefixes is caught.

`play` runs a script of requests through the solution and through the oracle, each
with its own clock and store, and compares per request: what `serve` returned, the
store operations it caused, and when it finished. The cost checks come last: bytes
moved between tiers and total serving time, against the specified policy's.

The solution's store is `Store`, a `SimTiers` behind the kit's `Guarded`: it may call
`compute`, `load`, `demote`, and `drop` and read `capacity`, `costs`, and `block_bytes`
(the two dicts are copies). The log, the violations, the totals, the clock, the
private index, and the test-only methods are refused.
"""

import asyncio

from learnai.judge import CheckFailure, case, check
from learnai.kit import Guarded, check_guard, guard

TIERS = ("gpu", "host", "disk")
BLOCK_BYTES = 1 << 20
MIB = 1 << 20

# Costs are multiples of 2**-6 seconds, so every sum of them is exact.
EXAMPLE_COSTS = {
    "compute_s": 0.25,
    "host_base_s": 0.125,
    "host_block_s": 0.0625,
    "disk_base_s": 0.5,
    "disk_block_s": 0.125,
}
# A host transfer of one block is slower than computing it (9/64 against 8/64); a
# disk transfer pays off from 6 blocks on; one of 4 blocks with DISK_TIE is a tie.
COSTS = {
    "compute_s": 8 / 64,
    "host_base_s": 8 / 64,
    "host_block_s": 1 / 64,
    "disk_base_s": 32 / 64,
    "disk_block_s": 2 / 64,
}
DISK_TIE = {**COSTS, "disk_base_s": 24 / 64}


class TierError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(f"{code}: {message}")
        self.code = code


class SimTiers:
    def __init__(self, sim, capacity, costs, block_size, names):
        self.sim = sim
        self._capacity = dict(capacity)
        self._costs = dict(costs)
        self.block_bytes = BLOCK_BYTES
        self._block_size = block_size
        self._names = names
        self._tier = {}  # key -> tier, for stored blocks
        self._prefix = {}  # key -> the tokens up to the end of the block
        self._key_of = {}  # prefix -> key
        self._parent = {}  # key -> parent key
        self._busy = False
        self.log = []  # (operation, ...) with blocks named by prefix
        self.violations = []
        self.bytes_moved = 0
        self.busy_s = 0.0

    # -- what a solution may call ---------------------------------------------------

    @property
    def capacity(self):
        return dict(self._capacity)

    @property
    def costs(self):
        return dict(self._costs)

    async def compute(self, key, parent, tokens):
        self._idle("compute")
        tokens = tuple(tokens)
        if len(tokens) != self._block_size or not all(type(t) is int for t in tokens):
            self._refuse("bad_tokens", f"compute needs the block's {self._block_size} tokens")
        if parent is not None and self._tier.get(parent) != "gpu":
            self._refuse(
                "missing_parent",
                f"compute of a block whose parent is not on the GPU ({self._where(parent)})",
            )
        prefix = (self._prefix[parent] if parent is not None else ()) + tokens
        if self._prefix.get(key, prefix) != prefix:
            self._refuse(
                "key_collision",
                f"one key is used for two different prefixes: {self.name(self._prefix[key])} "
                f"and {self.name(prefix)}",
            )
        if self._key_of.get(prefix, key) != key:
            self._refuse("second_key", f"block {self.name(prefix)} was given a second key")
        if key in self._tier:
            self._refuse(
                "duplicate",
                f"compute of block {self.name(prefix)}, which is still stored "
                f"({self._where(key)}); drop it first",
            )
        self._room("gpu", 1, f"compute of block {self.name(prefix)}")
        self._prefix[key], self._key_of[prefix], self._parent[key] = prefix, key, parent
        self._tier[key] = "gpu"
        self.log.append(("compute", self.name(prefix)))
        await self._take(self.costs["compute_s"])

    async def load(self, keys):
        self._idle("load")
        keys = list(keys)
        if not keys:
            self._refuse("empty_load", "load of no blocks")
        for key in keys:
            self._stored(key, "load")
        tiers = {self._tier[key] for key in keys}
        if len(tiers) != 1 or tiers & {"gpu"}:
            self._refuse(
                "wrong_tier",
                "a load must take blocks from one tier, the host or the disk; got "
                + ", ".join(f"{self.name(self._prefix[k])} ({self._tier[k]})" for k in keys),
            )
        if len(set(keys)) != len(keys):
            self._refuse("wrong_tier", "a block is listed twice in one load")
        tier = tiers.pop()
        self._room("gpu", len(keys), f"load of {len(keys)} blocks")
        arrived = set()
        for key in keys:
            parent = self._parent[key]
            if parent is not None and parent not in arrived and self._tier.get(parent) != "gpu":
                self._refuse(
                    "missing_parent",
                    f"load of block {self.name(self._prefix[key])}, whose parent is not on "
                    f"the GPU and not earlier in the same load ({self._where(parent)})",
                )
            arrived.add(key)
        for key in keys:
            self._tier[key] = "gpu"
        self.bytes_moved += self.block_bytes * len(keys)
        self.log.append(("load", tier, tuple(self.name(self._prefix[k]) for k in keys)))
        await self._take(self.costs[f"{tier}_base_s"] + self.costs[f"{tier}_block_s"] * len(keys))

    def demote(self, key, tier):
        self._idle("demote")
        self._stored(key, "demote")
        source = self._tier[key]
        if tier not in TIERS or TIERS.index(tier) <= TIERS.index(source):
            self._refuse(
                "wrong_tier",
                f"demote of block {self.name(self._prefix[key])} from {source} to {tier!r}: "
                "a block can only move down",
            )
        self._room(tier, 1, f"demote of block {self.name(self._prefix[key])} to {tier}")
        self._tier[key] = tier
        self.bytes_moved += self.block_bytes
        self.log.append(("demote", self.name(self._prefix[key]), source, tier))

    def drop(self, key):
        self._idle("drop")
        self._stored(key, "drop")
        self.log.append(("drop", self.name(self._prefix[key]), self._tier.pop(key)))

    # -- for the tests -----------------------------------------------------------------

    def name(self, prefix):
        return self._names(prefix)

    def contents(self):
        return {
            tier: sorted(self.name(self._prefix[k]) for k, t in self._tier.items() if t == tier)
            for tier in TIERS
        }

    def _where(self, key):
        if key not in self._prefix:
            return "the store has never seen that key"
        tier = self._tier.get(key)
        where = "is not stored" if tier is None else f"is on {tier}"
        return f"block {self.name(self._prefix[key])} {where}"

    def _stored(self, key, operation):
        if key not in self._tier:
            self._refuse("unknown_block", f"{operation} of a block that is not stored")

    def _room(self, tier, count, what):
        used = sum(1 for t in self._tier.values() if t == tier)
        if used + count > self.capacity[tier]:
            self._refuse(
                "tier_full",
                f"{what}: the {tier} tier has {self.capacity[tier] - used} free slots",
            )

    def _idle(self, operation):
        if self._busy:
            self._refuse("busy", f"{operation} while a load or a computation is running")

    def _refuse(self, code, message):
        error = TierError(code, message)
        self.violations.append(error)
        raise error

    async def _take(self, seconds):
        self._busy = True
        self.busy_s += seconds
        try:
            await asyncio.sleep(seconds)
        finally:
            self._busy = False


class Store(Guarded, SimTiers):
    """The store the solution gets: the four operations and the three settings."""

    label = "store"
    allowed = frozenset(
        {"capacity", "costs", "block_bytes", "compute", "load", "demote", "drop", "__class__"}
    )


# -- prompts made of named blocks -----------------------------------------------------


def blocks(letter, count, start=0, size=2):
    """Tokens of `count` blocks named letter+index: "A0", "A1", ..."""
    out = []
    for index in range(start, start + count):
        out += [ord(letter) * 1000 + index * 10 + j for j in range(size)]
    return out


def namer(size):
    def label(block):
        return f"{chr(block[0] // 1000)}{block[0] % 1000 // 10}"

    def name(prefix):
        labels = [label(prefix[i : i + size]) for i in range(0, len(prefix), size)]
        plain = all(
            lab == f"{labels[0][0]}{i}" for i, lab in enumerate(labels)
        )  # "A0/A1/A2" is just "A2"
        return labels[-1] if plain else "/".join(labels)

    return name


def show(operation):
    kind = operation[0]
    if kind == "load":
        return f"load {list(operation[2])} from {operation[1]}"
    if kind == "demote":
        return f"demote {operation[1]} {operation[2]} -> {operation[3]}"
    if kind == "drop":
        return f"drop {operation[1]} (was on {operation[2]})"
    return f"compute {operation[1]}"


def simulate(ctx, cls, script, capacity, costs, size, tiers=SimTiers):
    """Run the script on a fresh clock. Returns (store, per-step records, error)."""
    sim = ctx.simclock()
    store = tiers(sim, capacity, costs, size, namer(size))
    records = []
    failure = []

    async def driver():
        kv = cls(store, size)
        for kind, when, tokens in script:
            if sim.now() < when:
                await asyncio.sleep(when - sim.now())
            begin, mark = sim.now(), len(store.log)
            try:
                if kind == "serve":
                    result = await kv.serve(list(tokens))
                else:
                    result = kv.discard(list(tokens))
            except TierError as error:
                failure.append((len(records), error))
                return
            records.append(
                {
                    "result": result,
                    "ops": store.log[mark:],
                    "begin": begin,
                    "end": sim.now(),
                    "contents": store.contents(),
                }
            )

    sim.spawn(driver())
    sim.run()
    return store, records, failure[0] if failure else None


def play(ctx, script, *, capacity, costs=COSTS, size=2, expect=None):
    """Run `script` on the solution and the oracle and compare step by step.

    A step is ("serve", arrival_s, tokens) or ("discard", time_s, tokens). `expect`
    lists what each step returns according to the statement.
    """
    want_store, want, error = simulate(ctx, ctx.oracle.TieredKV, script, capacity, costs, size)
    if error is not None or len(want) != len(script):
        raise RuntimeError(f"authoring error: the oracle did not finish the script ({error})")
    if expect is not None and [r["result"] for r in want] != expect:
        raise RuntimeError(f"authoring error: the oracle returns {[r['result'] for r in want]}")
    guard(ctx)
    got_store, got, error = simulate(
        ctx, ctx.solution.TieredKV, script, capacity, costs, size, tiers=Store
    )
    check_guard()

    def history(upto):
        name = namer(size)
        lines = [f"capacity {capacity}, block_size {size}", f"costs {costs}"]
        for index, (kind, when, tokens) in enumerate(script[: upto + 1]):
            full = len(tokens) // size * size
            lines.append(
                f"step {index + 1}: {kind} at t={when:g}: blocks up to {name(tuple(tokens[:full])) if full else '(none)'}"
            )
            if index < len(got):
                lines.append(f"    returned {got[index]['result']!r} at t={got[index]['end']:g}")
                lines += [f"    {show(op)}" for op in got[index]["ops"]]
        return "\n".join(lines) + "\n"

    for index, (kind, _, _) in enumerate(script):
        step = f"step {index + 1} ({kind})"
        if error is not None and error[0] == index:
            raise CheckFailure(
                f"{step}: the store refused an operation: {error[1]}",
                history(index),
            )
        mine, theirs = got[index], want[index]
        expected = (
            f"expected: returned {theirs['result']!r} at t={theirs['end']:g}\n"
            + "".join(f"    {show(op)}\n" for op in theirs["ops"])
            + f"expected store afterwards: {theirs['contents']}\n"
        )
        if kind == "serve":
            keys = {"gpu", "host", "disk", "computed"}
            check.that(
                isinstance(mine["result"], dict)
                and set(mine["result"]) == keys
                and all(type(v) is int for v in mine["result"].values()),
                f"{step}: serve must return a dict with the int counts gpu, host, disk, computed",
                f"got {mine['result']!r}",
            )
        else:
            check.that(type(mine["result"]) is bool, f"{step}: discard must return True or False")
        for position, operation in enumerate(mine["ops"]):
            if position >= len(theirs["ops"]) or operation != theirs["ops"][position]:
                raise CheckFailure(
                    f"{step}: store operation {position + 1} of this step ({show(operation)}) "
                    "is not what the policy does at that point",
                    history(index) + expected,
                )
        check.that(
            len(mine["ops"]) == len(theirs["ops"]),
            f"{step}: the step ended after {len(mine['ops'])} store operations, "
            "before the policy was carried out in full",
            history(index) + expected,
        )
        check.that(
            mine["result"] == theirs["result"],
            f"{step}: returned {mine['result']!r}, which does not describe what was done",
            history(index) + expected,
        )
        check.that(
            mine["end"] == theirs["end"],
            f"{step}: finished at t={mine['end']:g}, which is not when the store's work ends",
            history(index) + expected,
        )
    check.that(not got_store.violations, "the store refused an operation (and it was caught)")
    check.at_most(
        "bytes_moved", got_store.bytes_moved / MIB, want_store.bytes_moved / MIB, unit="MiB"
    )
    check.at_most("serve_time", got_store.busy_s, want_store.busy_s, unit="s")
    return want_store, want


A, B, C, D = (lambda n, s=0, letter=letter: blocks(letter, n, s) for letter in "ABCD")


@case("the worked example: demote, load back, skip a full host tier, recompute", public=True)
def worked_example(ctx):
    script = [
        ("serve", 0.0, A(3)),
        ("serve", 1.0, B(3)),
        ("serve", 2.0, A(3)),
        ("serve", 3.0, B(3)),
    ]
    expect = [
        {"gpu": 0, "host": 0, "disk": 0, "computed": 3},
        {"gpu": 0, "host": 0, "disk": 0, "computed": 3},
        {"gpu": 1, "host": 2, "disk": 0, "computed": 0},
        {"gpu": 1, "host": 0, "disk": 0, "computed": 2},
    ]
    store, records = play(
        ctx,
        script,
        capacity={"gpu": 4, "host": 2, "disk": 4},
        costs=EXAMPLE_COSTS,
        expect=expect,
    )
    ends = [r["end"] for r in records]
    if ends != [0.75, 1.75, 2.25, 3.5] or store.bytes_moved != 8 * MIB:
        raise RuntimeError(f"authoring error: {ends}, {store.bytes_moved / MIB} MiB")
    if records[-1]["contents"] != {
        "gpu": ["A0", "B0", "B1", "B2"],
        "host": ["A1", "A2"],
        "disk": [],
    }:
        raise RuntimeError(f"authoring error: {records[-1]['contents']}")


@case("pressure cascades: GPU to host, host to disk, and off the end of the disk", public=True)
def cascade(ctx):
    # Four prompts of two blocks through a GPU of two: each serve pushes the previous
    # prompt down one tier, and the oldest falls off the disk.
    script = [("serve", float(i), blocks(letter, 2)) for i, letter in enumerate("ABCDE")]
    play(ctx, script, capacity={"gpu": 2, "host": 2, "disk": 2})


@case("a shared prefix stays hot: the deepest block of the oldest prompt goes first", public=True)
def shared_prefix(ctx):
    # S is a two-block system prompt shared by three conversations.
    S = blocks("S", 2)
    script = [
        ("serve", 0.0, S + A(2, 2)),
        ("serve", 1.0, S + B(2, 2)),
        ("serve", 2.0, S + C(2, 2)),
        ("serve", 3.0, S + A(2, 2)),
        ("serve", 4.0, S + B(2, 2) + [7]),  # one token past the last full block
    ]
    # The last two serves find one block on the host (loaded) and one on disk (a
    # transfer of one block from disk is slower than computing it).
    _, records = play(ctx, script, capacity={"gpu": 5, "host": 2, "disk": 3}, costs=EXAMPLE_COSTS)
    if records[3]["result"] != {"gpu": 2, "host": 1, "disk": 0, "computed": 1}:
        raise RuntimeError(f"authoring error: {records[3]['result']}")


@case("load or recompute: a transfer's base cost against the blocks it brings", public=True)
def load_or_recompute(ctx):
    # Computing a block takes 8/64 s. A host transfer of n blocks takes (8 + n)/64 s,
    # a disk transfer (32 + 2n)/64 s.
    capacity = {"gpu": 6, "host": 2, "disk": 8}
    script = [
        ("serve", 0.0, A(6)),
        ("serve", 1.0, B(5)),  # A1, A2 end up on the host, A3..A5 on disk
        ("serve", 2.0, A(1)),  # on the GPU: nothing to do
        ("serve", 3.0, A(6)),  # the host run of 2 is loaded; the disk run of 3 is not
        ("serve", 4.0, B(6)),  # 5 blocks on disk: 42/64 to load, 40/64 to compute
        ("serve", 5.0, A(6)),
    ]
    play(ctx, script, capacity=capacity)


@case("requests that queue: each serve starts when the previous one ends", public=True)
def queued(ctx):
    script = [
        ("serve", 0.0, A(4)),
        ("serve", 0.0, B(4)),
        ("serve", 0.125, A(4)),
        ("serve", 5.0, B(4)),
        ("serve", 5.0, []),
        ("serve", 5.0, [1]),
    ]
    play(ctx, script, capacity={"gpu": 4, "host": 4, "disk": 4})


@case("a short host run rides along with a long disk run")
def combined_runs(ctx):
    # One block on the host (9/64 s to load, 8/64 to compute) in front of six on disk
    # (44/64 against 48/64): loading both costs 53/64, computing all seven 56/64.
    capacity = {"gpu": 7, "host": 1, "disk": 8}
    script = [
        ("serve", 0.0, A(7)),
        ("serve", 1.0, B(7)),
        ("serve", 2.0, A(7)),
        ("serve", 3.0, B(6)),
        ("serve", 4.0, A(7)),
    ]
    _, records = play(ctx, script, capacity=capacity)
    if records[2]["result"] != {"gpu": 0, "host": 1, "disk": 6, "computed": 0}:
        raise RuntimeError(f"authoring error: {records[2]['result']}")


@case("a tie between loading and computing goes to loading")
def tie_prefers_reuse(ctx):
    capacity = {"gpu": 4, "host": 4, "disk": 8}
    script = [
        ("serve", 0.0, A(4)),
        ("serve", 1.0, B(4)),
        ("serve", 2.0, C(4)),  # A is on disk now, B on the host
        ("serve", 3.0, A(4)),  # 4 blocks from disk: 24/64 + 8/64 = 4 * 8/64
        ("serve", 4.0, B(3)),
    ]
    _, records = play(ctx, script, capacity=capacity, costs=DISK_TIE)
    if records[3]["result"] != {"gpu": 0, "host": 0, "disk": 4, "computed": 0}:
        raise RuntimeError(f"authoring error: {records[3]['result']}")


@case("a discarded block: its descendants are recomputed, never loaded")
def discarded_ancestor(ctx):
    capacity = {"gpu": 5, "host": 5, "disk": 5}
    script = [
        ("serve", 0.0, A(5)),
        ("serve", 1.0, B(5)),  # A is on the host
        ("discard", 1.5, A(3)),  # A2 goes; A3 and A4 stay behind it, unreachable
        ("discard", 1.5, A(3)),  # already gone
        ("discard", 1.5, [5]),  # no full block
        ("serve", 2.0, A(5)),  # A0 and A1 are loaded; A2, A3, A4 are computed
        ("discard", 3.0, A(4)),  # A3, on the GPU
        ("serve", 4.0, A(5)),  # A4 is on the GPU and is computed again all the same
        ("serve", 5.0, B(5)),
        ("discard", 6.0, B(1)),  # the head of a chain
        ("serve", 7.0, B(5)),
    ]
    _, records = play(ctx, script, capacity=capacity)
    discards = [r["result"] for r, step in zip(records, script, strict=True) if step[0] != "serve"]
    if discards != [True, False, False, True, True]:
        raise RuntimeError(f"authoring error: {discards}")
    if records[5]["result"] != {"gpu": 0, "host": 2, "disk": 0, "computed": 3}:
        raise RuntimeError(f"authoring error: {records[5]['result']}")


@case("the same tokens after different prefixes are different blocks")
def same_tokens_other_prefix(ctx):
    X = blocks("X", 2)
    script = [
        ("serve", 0.0, A(1) + X),
        ("serve", 1.0, B(1) + X),  # X0, X1 again, but after B0: nothing to reuse
        ("serve", 2.0, X),
        ("serve", 3.0, A(1) + X),
        ("serve", 4.0, X + A(1)),
    ]
    play(ctx, script, capacity={"gpu": 6, "host": 3, "disk": 3})


@case("a hit refreshes a block: what was just used is not the next to go")
def hits_refresh(ctx):
    script = [
        ("serve", 0.0, A(2)),
        ("serve", 1.0, B(2)),
        ("serve", 2.0, A(2)),  # all on the GPU: only the recency changes
        ("serve", 3.0, C(2)),  # B must go, not A
        ("serve", 4.0, A(2)),
        ("serve", 5.0, D(4)),
    ]
    play(ctx, script, capacity={"gpu": 4, "host": 2, "disk": 2})


@case("block_size 4 and a disk tier of one block")
def other_sizes(ctx):
    def prompt(letter, n):
        return blocks(letter, n, size=4)

    script = [
        ("serve", 0.0, prompt("A", 3) + [1, 2, 3]),
        ("serve", 1.0, prompt("B", 2)),
        ("serve", 2.0, prompt("C", 3)),
        ("serve", 3.0, prompt("A", 3)),
        ("serve", 4.0, prompt("B", 3)),
    ]
    play(ctx, script, capacity={"gpu": 3, "host": 2, "disk": 1}, size=4)


def random_script(rng, steps, gpu):
    """Conversations that share system prompts, grow, come back, and are discarded."""
    systems = {letter: blocks(letter, int(rng.integers(1, 3))) for letter in "ST"}
    prompts = []
    for letter in "ABCDEFGH":
        system = systems["S" if rng.random() < 0.6 else "T"]
        depth = len(system) // 2
        prompts.append(system + blocks(letter, int(rng.integers(1, gpu - depth + 1)), depth))
    script, now = [], 0.0
    for _ in range(steps):
        now += float(rng.integers(0, 4)) / 4
        tokens = prompts[int(rng.integers(0, len(prompts)))]
        cut = int(rng.integers(1, len(tokens) // 2 + 1)) * 2
        if rng.random() < 0.12:
            script.append(("discard", now, tokens[:cut]))
        else:
            tail = [9] if rng.random() < 0.3 else []
            script.append(
                ("serve", now, tokens[: cut if rng.random() < 0.3 else len(tokens)] + tail)
            )
    return script


@case("random conversations with shared system prompts, three tier layouts")
def random_workloads(ctx):
    layouts = [
        ({"gpu": 6, "host": 4, "disk": 8}, COSTS),
        ({"gpu": 5, "host": 2, "disk": 3}, DISK_TIE),
        ({"gpu": 8, "host": 6, "disk": 12}, EXAMPLE_COSTS),
    ]
    for trial, (capacity, costs) in enumerate(layouts):
        script = random_script(ctx.child_rng(trial), 60, capacity["gpu"])
        play(ctx, script, capacity=capacity, costs=costs)
