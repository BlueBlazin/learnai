"""Tests for capstone-continuous-batching-engine.

Stage 1 exercises `BlockAllocator` alone: scripted calls with literal results, random
workloads against the oracle, and invariants computed here (reference counts equal
table membership, nothing leaks, a refused call changes nothing).

Stages 2 to 5 run `Engine` on a `SimEngine` whose clock counts milliseconds. The
kit engine has no memory pool of its own: the solution's allocator is the memory,
and `Watched`, a small subclass of `SimEngine` defined here, checks it against the
engine's own token counts at every step (who holds blocks, how many, reference
counts, no leak). Schedules are compared with literal steps on the statement's
examples and with the oracle's trace on seeded workloads; the costs of the policy
(recomputed tokens, prefilled tokens, peak blocks) have limits.

The solution's `Engine` gets the `Watched` engine as `Handed`, an object behind the
kit's `Guarded` that passes on the calls, the queries, and the limits of a `SimEngine`.
The engine itself, and with it the counters, the clock and its trace, the violations,
and what the audit recorded, cannot be reached through it.
"""

from itertools import pairwise

from learnai.judge import case, check
from learnai.kit import (
    CostModel,
    Guarded,
    ProtocolViolation,
    SimEngine,
    SimRequest,
    check_guard,
    guard,
    workloads,
)

S1 = "1: block allocator"
S2 = "2: continuous batching"
S3 = "3: chunked prefill"
S4 = "4: priorities and preemption"
S5 = "5: prefix sharing"

# A step over P prompt tokens and D decoded sequences takes 8 + 0.25 * P + D milliseconds.
COST = CostModel(base_s=8, prefill_s_per_token=0.25, decode_s_per_seq=1)
AMPLE = 4096  # a token budget and a pool that never bind (stage 2)


# -- stage 1 helpers ---------------------------------------------------------------------


def audit(alloc, n_blocks, block_size, tokens, history):
    """Invariants of an allocator whose live sequences hold `tokens[seq_id]` tokens."""
    detail = "calls so far:\n" + "\n".join(history[-25:]) + "\n"
    held = {}
    for seq_id, count in tokens.items():
        table = alloc.block_table(seq_id)
        check.that(
            isinstance(table, list) and all(type(b) is int for b in table),
            "block_table: did not return a list of ints",
            detail,
        )
        check.that(
            len(table) == -(-count // block_size),
            f"a block table has {len(table)} entries, expected {-(-count // block_size)} "
            "(ceil(tokens / block_size))",
            detail + f"sequence {seq_id!r}: {count} tokens, table {table}\n",
        )
        check.that(
            len(set(table)) == len(table) and all(0 <= b < n_blocks for b in table),
            "a block table repeats a block or names one outside the pool",
            detail + f"sequence {seq_id!r}: table {table}\n",
        )
        for b in table:
            held[b] = held.get(b, 0) + 1
    for b in range(n_blocks):
        refs = alloc.ref_count(b)
        check.that(
            refs == held.get(b, 0),
            f"ref_count({b}) is {refs!r}, expected {held.get(b, 0)} (the number of block tables "
            "that hold the block)",
            detail + f"block {b}: ref_count {refs}, in {held.get(b, 0)} tables\n",
        )
    free = alloc.free_blocks()
    check.that(
        free == n_blocks - len(held),
        f"free_blocks() is {free!r}, expected {n_blocks - len(held)} (the blocks in no block "
        "table)",
        detail + f"free_blocks() = {free}, {len(held)} of {n_blocks} blocks held\n",
    )


class Script:
    """Plays calls on the solution's allocator and audits it after every one."""

    def __init__(self, ctx, n_blocks, block_size):
        self.module = ctx.solution
        self.alloc = ctx.solution.BlockAllocator(n_blocks, block_size)
        self.n_blocks, self.block_size = n_blocks, block_size
        self.tokens = {}
        self.history = [f"BlockAllocator(n_blocks={n_blocks}, block_size={block_size})"]

    def _done(self, text):
        self.history.append(text)
        audit(self.alloc, self.n_blocks, self.block_size, self.tokens, self.history)

    def _detail(self, text, expected):
        return (
            "calls so far:\n" + "\n".join([*self.history[-25:], text]) + f"\nexpected: {expected}\n"
        )

    def append(self, seq_id, n, expected):
        text = f"append({seq_id!r}, {n})"
        if expected is OUT:
            check.raises(self.module.OutOfBlocks, self.alloc.append, seq_id, n, label=text)
            self._done(text + "  -> OutOfBlocks")
            return
        got = self.alloc.append(seq_id, n)
        ok = (
            isinstance(got, tuple)
            and len(got) == 2
            and list(got[0]) == expected[0]
            and [tuple(pair) for pair in got[1]] == expected[1]
        )
        check.that(ok, f"{text}: returned {got!r}, expected {tuple(expected)!r}",
                   self._detail(text, expected))  # fmt: skip
        self.tokens[seq_id] = self.tokens.get(seq_id, 0) + n
        self._done(f"{text}  -> {got!r}")

    def fork(self, parent, child, n_tokens=None):
        if n_tokens is None:
            text = f"fork({parent!r}, {child!r})"
            result = self.alloc.fork(parent, child)
        else:
            text = f"fork({parent!r}, {child!r}, {n_tokens})"
            result = self.alloc.fork(parent, child, n_tokens)
        check.that(result is None, f"{text}: must return None")
        self.tokens[child] = self.tokens[parent] if n_tokens is None else n_tokens
        self._done(text)

    def free(self, seq_id):
        result = self.alloc.free(seq_id)
        check.that(result is None, f"free({seq_id!r}): must return None")
        del self.tokens[seq_id]
        self._done(f"free({seq_id!r})")

    def table(self, seq_id, expected):
        got = self.alloc.block_table(seq_id)
        check.that(got == expected, f"block_table({seq_id!r}): got {got!r}, expected {expected!r}",
                   self._detail(f"block_table({seq_id!r})", expected))  # fmt: skip

    def free_blocks(self, expected):
        got = self.alloc.free_blocks()
        check.that(got == expected, f"free_blocks(): got {got!r}, expected {expected!r}",
                   self._detail("free_blocks()", expected))  # fmt: skip

    def refs(self, block, expected):
        got = self.alloc.ref_count(block)
        check.that(got == expected, f"ref_count({block}): got {got!r}, expected {expected!r}",
                   self._detail(f"ref_count({block})", expected))  # fmt: skip


OUT = object()


def random_ops(ctx, rng, n_blocks, block_size, n_ops, *, prefix_forks):
    """A random workload on the solution and the oracle; every result must agree."""
    mine = ctx.solution.BlockAllocator(n_blocks, block_size)
    theirs = ctx.oracle.BlockAllocator(n_blocks, block_size)
    tokens, history, next_id = {}, [f"BlockAllocator({n_blocks}, {block_size})"], 0
    for _ in range(n_ops):
        live = sorted(tokens)
        roll = rng.random()
        if not live or roll < 0.15:
            op, args = "append", (next_id, int(rng.integers(1, 3 * block_size)))
            next_id += 1
        elif roll < 0.55:
            op, args = "append", (int(rng.choice(live)), int(rng.integers(1, 2 * block_size)))
        elif roll < 0.8:
            parent = int(rng.choice(live))
            args = (parent, next_id)
            if prefix_forks and rng.random() < 0.6:
                args = (parent, next_id, int(rng.integers(1, tokens[parent] + 1)))
            op = "fork"
            next_id += 1
        else:
            op, args = "free", (int(rng.choice(live)),)
        text = f"{op}({', '.join(map(repr, args))})"
        try:
            want = getattr(theirs, op)(*args)
        except ctx.oracle.OutOfBlocks:
            check.raises(ctx.solution.OutOfBlocks, getattr(mine, op), *args, label=text)
            history.append(text + "  -> OutOfBlocks")
        else:
            got = getattr(mine, op)(*args)
            if op == "append":
                got = (list(got[0]), [tuple(pair) for pair in got[1]])
                tokens[args[0]] = tokens.get(args[0], 0) + args[1]
            elif op == "fork":
                tokens[args[1]] = args[2] if len(args) == 3 else tokens[args[0]]
            else:
                del tokens[args[0]]
            check.that(
                got == want,
                f"{text}: returned {got!r}, expected {want!r}",
                "calls so far:\n" + "\n".join(history[-25:]) + f"\n{text}\n"
                f"expected: {want!r}\ngot:      {got!r}\n",
            )
            history.append(f"{text}  -> {got!r}")
        audit(mine, n_blocks, block_size, tokens, history)
        for seq_id in tokens:
            table, due = mine.block_table(seq_id), theirs.block_table(seq_id)
            check.that(
                table == due,
                f"block_table({seq_id!r}): got {table!r}, expected {due!r}",
                "calls so far:\n" + "\n".join(history[-25:]) + f"\nsequence {seq_id!r}: expected "
                f"{due}, got {table}\n",
            )


# -- stages 2 to 5: the engine under watch ------------------------------------------------


class Watched(SimEngine):
    """A SimEngine that audits the solution's allocator whenever a step is planned.

    The kit engine counts tokens; the allocator under test owns the blocks. At each
    `step` call the two must agree: a sequence that holds memory has a block table of
    ceil(tokens / block_size) blocks, where tokens is what it has in the KV cache
    plus what its admission reserved (its remaining prefill) plus this step's decode;
    a sequence that holds none (preempted and not readmitted) has no table.
    """

    def __init__(self, sim, n_blocks, block_size, max_batch_tokens, max_running):
        super().__init__(sim, COST, max_batch_tokens=max_batch_tokens, block_tokens=block_size)
        self.n_blocks, self.block_size, self.max_running = n_blocks, block_size, max_running
        self.allocator = None  # set by simulate()
        self.holders = set()
        self.problems = []  # (message, extra detail)
        self.peak_blocks = 0
        self.copies = 0

    def admit(self, request, **options):
        super().admit(request, **options)
        self.holders.add(request.id)

    def preempt(self, seq_id):
        super().preempt(seq_id)
        self.holders.discard(seq_id)

    async def step(self, prefill=(), decode=()):
        prefill, decode = tuple(prefill), tuple(decode)
        if not self.busy:
            try:
                self._audit(prefill, decode)
            except (KeyError, IndexError, TypeError, ValueError, AttributeError) as error:
                # A broken allocator is reported after the run, with the timeline.
                self._problem(f"reading the allocator raised {type(error).__name__}", f"{error}")
        return await super().step(prefill=prefill, decode=decode)

    def _problem(self, message, extra=""):
        if not self.problems:
            where = f"at t={self.sim.now():g} ms, before step {self.counters.steps}: "
            self.problems.append((message, f"{where}{extra}\n"))

    def _audit(self, prefill, decode):
        alloc = self.allocator
        live = set(self._seqs)
        self.holders = (self.holders | {i for i, _ in prefill}) & live
        if len(self.holders) > self.max_running:
            self._problem("more than max_running sequences hold memory",
                          f"holders: {sorted(self.holders, key=repr)}")  # fmt: skip
        held = {}
        for seq_id, seq in self._seqs.items():
            if seq_id not in self.holders:
                try:
                    table = alloc.block_table(seq_id)
                except KeyError:
                    continue
                self._problem("a preempted sequence still has a block table",
                              f"sequence {seq_id!r}: {table}")  # fmt: skip
                continue
            tokens = seq.kv_tokens + seq.remaining_prefill + (seq_id in decode)
            table = alloc.block_table(seq_id)
            if len(table) != -(-tokens // self.block_size):
                self._problem(
                    f"a sequence's block table has {len(table)} blocks, expected "
                    f"{-(-tokens // self.block_size)} for the {tokens} tokens it holds",
                    f"sequence {seq_id!r} needs blocks for {tokens} tokens (KV "
                    f"{seq.kv_tokens}, prefill to come {seq.remaining_prefill}, decoding "
                    f"now: {seq_id in decode}); its table is {table}",
                )
            if len(set(table)) != len(table):
                self._problem("a block table lists a block twice", f"{seq_id!r}: {table}")
            for b in table:
                held[b] = held.get(b, 0) + 1
        for b, count in held.items():
            if not 0 <= b < self.n_blocks or alloc.ref_count(b) != count:
                self._problem(
                    "a block's reference count is not the number of sequences that hold it",
                    f"block {b}: held by {count} live sequences, ref_count "
                    f"{alloc.ref_count(b) if 0 <= b < self.n_blocks else 'n/a'}",
                )
        if alloc.free_blocks() != self.n_blocks - len(held):
            self._problem(
                "blocks are neither free nor held by a live sequence (a leak), or both",
                f"free_blocks() = {alloc.free_blocks()}, live sequences hold {len(held)} "
                f"distinct blocks, the pool has {self.n_blocks}",
            )
        self.peak_blocks = max(self.peak_blocks, len(held))


class Handed(Guarded):
    """The `sim_engine` a solution gets: what a `SimEngine` offers, passed on to `engine`.

    The engine stays a plain object, because the kit's own bookkeeping reads its
    attributes several hundred thousand times in a run.
    """

    label = "sim_engine"
    refused = Guarded.refused | {"engine", "sim", "violations"}

    def __init__(self, engine):
        self.engine = engine

    def admit(self, request, **options):
        return self.engine.admit(request, **options)

    async def step(self, prefill=(), decode=()):
        return await self.engine.step(prefill=prefill, decode=decode)


def passed_on(name):
    def method(self, seq_id):
        return getattr(self.engine, name)(seq_id)

    return method


for _name in ("preempt", "release", "remaining_prefill", "generated", "kv_tokens", "blocks"):
    setattr(Handed, _name, passed_on(_name))
for _name in ("name", "cost", "max_batch_tokens", "max_seqs", "kv_blocks", "block_tokens",
              "busy", "busy_s", "live", "free_blocks"):  # fmt: skip
    setattr(Handed, _name, property(lambda self, _name=_name: getattr(self.engine, _name)))


class Run:
    def __init__(self, sim, engine, requests, config):
        self.sim, self.engine, self.requests, self.config = sim, engine, requests, config

    @property
    def steps(self):
        """[(start time, prefill, decode)]."""
        return [
            (e.time, tuple(e["prefill"]), tuple(e["decode"])) for e in self.sim.trace.of("step")
        ]

    @property
    def preempts(self):
        return [(e.time, e["id"]) for e in self.sim.trace.of("preempt")]

    @property
    def admits(self):
        return [(e.time, e["id"], e.fields.get("cached", 0)) for e in self.sim.trace.of("admit")]

    def timeline(self):
        lines = [self.config, "a step takes 8 + 0.25 * P + D ms", "timeline:"]
        kinds = ("arrive", "admit", "step", "finish", "preempt", "release", "violation")
        shown = [e for e in self.sim.trace.events if e.kind in kinds]
        for e in shown[:120]:
            if e.kind == "arrive":
                r = e["request"]
                what = f"arrive   {r.id!r} (prompt {r.prompt_tokens}, output {r.output_tokens}"
                what += f", priority {r.priority})" if r.priority else ")"
            elif e.kind == "admit":
                cached = e.fields.get("cached", 0)
                if not cached:
                    continue
                what = f"admit    {e['id']!r} sharing {cached} tokens"
            elif e.kind == "step":
                what = f"step {e['index']}: prefill={list(e['prefill'])} decode={list(e['decode'])}"
                what += f"  ({e['duration']:g} ms)"
            elif e.kind == "violation":
                what = f"VIOLATION {e['code']}: {e['message']}"
            elif e.kind == "preempt":
                what = f"preempt  {e['id']!r} (after {e['generated']} tokens)"
            else:
                what = f"{e.kind:<8} {e['id']!r}"
            lines.append(f"  t={e.time:>8g}  {what}")
        if len(shown) > 120:
            lines.append(f"  ... {len(shown) - 120} more events")
        return "\n".join(lines) + "\n"


def simulate(ctx, module, requests, *, n_blocks=AMPLE, block_size=16, max_batch_tokens=AMPLE,
             **options):  # fmt: skip
    """Run `module.Engine` on the workload; fail on a protocol violation or a bad audit."""
    sim = ctx.simclock()
    max_running = options.get("max_running", 8)
    watched = Watched(sim, n_blocks, block_size, max_batch_tokens, max_running)
    config = (
        f"Engine(sim_engine, n_blocks={n_blocks}, block_size={block_size}, "
        f"max_batch_tokens={max_batch_tokens}"
        + "".join(f", {k}={v}" for k, v in options.items())
        + ")"
    )
    run = Run(sim, watched, requests, config)
    guard(ctx)  # the solution's code runs from here to the end of `sim.run`
    engine = module.Engine(Handed(watched), n_blocks, block_size, max_batch_tokens, **options)
    check.that(
        hasattr(engine, "allocator")
        and all(hasattr(engine.allocator, m) for m in ("block_table", "ref_count", "free_blocks")),
        "Engine has no `allocator` attribute holding its BlockAllocator",
    )
    watched.allocator = engine.allocator
    workloads.drive(sim, requests, engine.submit)
    sim.spawn(engine.run())
    serial = sum(
        COST.step_s(r.prompt_tokens, 0) + r.output_tokens * COST.step_s(0, 1) for r in requests
    )
    try:
        # Generous: every request alone, many times over, with room for recomputation.
        sim.run(until=requests[-1].arrival_s + 20 * serial + 1000)
    except ProtocolViolation:
        pass  # reported below, with the timeline
    check_guard()
    if watched.violations:
        first = watched.violations[0]
        check.that(False, f"the engine broke the SimEngine protocol ({first.code})",
                   f"{first.code}: {first}\n" + run.timeline())  # fmt: skip
    if watched.problems:
        message, extra = watched.problems[0]
        check.that(False, message, extra + run.timeline())
    return run


def check_rules(run):
    """What the statement promises for every schedule."""
    detail = run.timeline()
    trace = run.sim.trace
    finished = {e["id"] for e in trace.of("finish")}
    missing = [r.id for r in run.requests if r.id not in finished]
    check.that(not missing, f"{len(missing)} of {len(run.requests)} requests never finished",
               detail)  # fmt: skip
    check.that(not trace.of("release"), "a sequence was released; preemption keeps it", detail)
    produced = {}
    for e in trace.of("token"):
        produced.setdefault(e["id"], []).append(e["index"])
    for request in run.requests:
        check.that(
            produced.get(request.id, []) == list(range(request.output_tokens)),
            "a request's tokens were not produced exactly once each, in order",
            detail,
        )
    alloc = run.engine.allocator
    check.that(
        alloc.free_blocks() == run.engine.n_blocks,
        "blocks are still allocated after every request has finished (a leak)",
        detail + f"free_blocks() = {alloc.free_blocks()} of {run.engine.n_blocks}\n",
    )


def check_steps(run, expected, label):
    """Compare the steps with a literal schedule [(start, prefill, decode)]."""
    got = run.steps
    for k, want in enumerate(expected):
        want = (want[0], tuple(want[1]), tuple(want[2]))
        if k >= len(got) or got[k] != want:
            mine = "no such step" if k >= len(got) else show_step(got[k])
            check.that(False, f"{label}: step {k} is not the specified step",
                       f"expected: {show_step(want)}\ngot:      {mine}\n" + run.timeline())  # fmt: skip
    check.that(len(got) == len(expected), f"{label}: the engine ran {len(got)} steps, expected "
               f"{len(expected)}", run.timeline())  # fmt: skip


def show_step(step):
    return f"t={step[0]:g}  prefill={list(step[1])}  decode={list(step[2])}"


def check_against_oracle(ctx, requests, label, **config):
    want = simulate(ctx, ctx.oracle, requests, **config)
    got = simulate(ctx, ctx.solution, requests, **config)
    check_rules(got)
    for kind, mine, theirs, show in (
        ("step", got.steps, want.steps, show_step),
        ("preemption", got.preempts, want.preempts, lambda p: f"t={p[0]:g}  sequence {p[1]!r}"),
        ("admission", got.admits, want.admits,
         lambda a: f"t={a[0]:g}  sequence {a[1]!r} sharing {a[2]} tokens"),
    ):  # fmt: skip
        for k, (a, b) in enumerate(zip(mine, theirs, strict=False)):
            if a != b:
                check.that(False, f"{label}: {kind} {k} is not the specified one",
                           f"expected: {show(b)}\ngot:      {show(a)}\n" + got.timeline())  # fmt: skip
        check.that(len(mine) == len(theirs), f"{label}: got {len(mine)} {kind}s, expected "
                   f"{len(theirs)}", got.timeline())  # fmt: skip
    return got


# -- stage 1: the allocator ---------------------------------------------------------------


@case("the allocator example: append, fork, copy on write, free", public=True, stage=S1)
def allocator_example(ctx):
    s = Script(ctx, n_blocks=6, block_size=4)
    s.append("a", 6, ([0, 1], []))
    s.fork("a", "b")
    s.refs(0, 2)
    s.append("b", 1, ([2], [(1, 2)]))  # block 1 is shared and partly filled: b copies it
    s.table("b", [0, 2])
    s.refs(1, 1)
    s.append("a", 1, ([], []))  # a is the only holder of block 1 now: in place
    s.append("b", 4, ([3], []))
    s.fork("b", "c")
    s.free("b")  # c holds every block b held: nothing returns to the pool
    s.free_blocks(2)
    s.append("c", 2, ([4], []))  # block 3 (3 of 4) is c's alone
    s.free("a")
    s.free_blocks(2)  # block 1 returned, block 0 is still c's
    s.table("c", [0, 2, 3, 4])
    s.free("c")
    s.free_blocks(6)


@case("blocks are taken lowest id first, and only when the last block is full", public=True,
      stage=S1)  # fmt: skip
def lowest_id_first(ctx):
    s = Script(ctx, n_blocks=6, block_size=4)
    s.append("a", 5, ([0, 1], []))
    s.append("b", 3, ([2], []))
    s.append("a", 3, ([], []))  # fits in block 1
    s.append("a", 1, ([3], []))
    s.free("b")
    s.append("c", 9, ([2, 4, 5], []))  # the freed block 2 comes first
    s.table("a", [0, 1, 3])
    s.free("a")
    s.append("d", 4, ([0], []))


@case("a refused append raises OutOfBlocks and changes nothing", public=True, stage=S1)
def refused_append(ctx):
    # Rejects leaks_on_refused_append: the blocks it took before giving up are gone.
    s = Script(ctx, n_blocks=4, block_size=4)
    s.append("a", 9, ([0, 1, 2], []))
    s.append("b", 6, OUT)  # needs 2 blocks, 1 is free; b must not become live
    s.free_blocks(1)
    check.raises(KeyError, s.alloc.block_table, "b", label="block_table of a refused sequence")
    s.append("a", 8, OUT)  # 3 tokens would fit in block 2, but an append is all or nothing
    s.table("a", [0, 1, 2])
    s.append("a", 3, ([], []))
    s.fork("a", "c")
    s.append("c", 1, ([3], []))
    s.append("a", 1, OUT)
    s.free("c")
    s.append("a", 1, ([3], []))


@case("fork errors, free errors, and tables that are copies", stage=S1)
def allocator_errors(ctx):
    s = Script(ctx, n_blocks=8, block_size=4)
    s.append("a", 6, ([0, 1], []))
    s.fork("a", "b")
    check.raises(ValueError, s.alloc.fork, "a", "b", label="fork onto a live child")
    check.raises(ValueError, s.alloc.fork, "a", "a", label="fork onto itself")
    check.raises(KeyError, s.alloc.fork, "nobody", "c", label="fork of an unknown parent")
    check.raises(KeyError, s.alloc.free, "nobody", label="free of an unknown sequence")
    check.raises(KeyError, s.alloc.block_table, "nobody", label="block_table of an unknown id")
    audit(s.alloc, 8, 4, s.tokens, s.history)
    s.alloc.block_table("a").append(7)  # the caller's list, not the allocator's
    s.alloc.block_table("b").clear()
    s.table("a", [0, 1])
    s.table("b", [0, 1])
    s.free("a")
    check.raises(KeyError, s.alloc.free, "a", label="free of a freed sequence")
    s.append("b", 1, ([], []))  # b is alone now: no copy
    s.free("b")
    s.free_blocks(8)


@case("siblings: only the first writers of a shared block copy it", stage=S1)
def siblings_copy_once(ctx):
    # Rejects no_copy_on_shared_block and fork_aliases_table.
    s = Script(ctx, n_blocks=16, block_size=4)
    s.append("p", 10, ([0, 1, 2], []))
    for child in ("c1", "c2", "c3"):
        s.fork("p", child)
    s.refs(2, 4)
    s.append("p", 1, ([3], [(2, 3)]))
    s.append("c1", 3, ([4, 5], [(2, 4)]))
    s.append("c2", 2, ([6], [(2, 6)]))
    s.append("c3", 2, ([], []))  # the last holder of block 2 writes in place
    s.table("c3", [0, 1, 2])
    s.refs(0, 4)
    s.free("p")
    s.free("c1")
    s.free_blocks(12)
    s.free("c2")
    s.free("c3")
    s.free_blocks(16)


@case("random workloads of appends, whole forks, and frees agree with the oracle", stage=S1)
def allocator_random(ctx):
    def prop(rng):
        n_blocks = int(rng.choice([6, 12, 40]))
        block_size = int(rng.choice([1, 3, 4, 16]))
        random_ops(ctx, rng, n_blocks, block_size, 80, prefix_forks=False)

    check.forall(prop, trials=12)


# -- stage 2: continuous batching ----------------------------------------------------------

EXAMPLE_2 = [(0, 8, 4), (0, 16, 2), (5, 8, 3), (45, 4, 2), (50, 12, 1), (60, 4, 2)]


@case("the stage 2 example: six requests, two slots", public=True, stage=S2)
def batching_example(ctx):
    run = simulate(ctx, ctx.solution, workloads.replay(EXAMPLE_2), max_running=2)
    check_rules(run)
    check_steps(
        run,
        [
            (0, [(0, 8), (1, 16)], []),
            (14, [], [0, 1]),
            (24, [(2, 8)], [0]),
            (35, [], [0, 2]),
            (45, [(3, 4)], [2]),
            (55, [(4, 12)], [3]),
            (67, [(5, 4)], []),
            (76, [], [5]),
        ],
        "the stage 2 example",
    )
    latency = run.sim.trace.latencies("arrive", "finish")
    check.equal(latency, [45.0, 24.0, 50.0, 22.0, 17.0, 25.0], label="end-to-end latencies")


@case("the engine idles between requests and wakes on a submission", public=True, stage=S2)
def idle_and_wake(ctx):
    requests = workloads.replay([(0, 4, 2), (100, 4, 1), (100, 8, 2), (300, 2, 1)])
    run = simulate(ctx, ctx.solution, requests, max_running=4)
    check_rules(run)
    check_steps(
        run,
        [(0, [(0, 4)], []), (9, [], [0]), (100, [(1, 4), (2, 8)], []), (111, [], [2]),
         (300, [(3, 2)], [])],
        "idle and wake",
    )  # fmt: skip


@case("the allocator follows the engine: blocks taken at admission, returned at the finish",
      public=True, stage=S2)  # fmt: skip
def allocator_follows(ctx):
    # Rejects blocks_not_freed_at_finish (the audit before the third step finds a leak).
    requests = workloads.replay([(0, 9, 2), (0, 4, 3), (30, 5, 2)])
    run = simulate(ctx, ctx.solution, requests, n_blocks=8, block_size=4, max_running=2)
    check_rules(run)
    check.at_most("peak_blocks", run.engine.peak_blocks, 5, unit="blocks")


@case("a slot freed at the end of a step is filled in the next step", stage=S2)
def slot_refilled_at_once(ctx):
    # Rejects slot_idles_one_step.
    requests = workloads.replay([(0, 8, 2), (0, 8, 3), (1, 8, 2), (1, 4, 2), (19, 4, 1)])
    run = simulate(ctx, ctx.solution, requests, max_running=2)
    check_rules(run)
    check_steps(
        run,
        [
            (0, [(0, 8), (1, 8)], []),
            (12, [], [0, 1]),
            (22, [(2, 8)], [1]),
            (33, [(3, 4)], [2]),
            (43, [(4, 4)], [3]),
        ],
        "two slots, five requests",
    )


@case("bursts and Poisson arrivals: the steps equal the oracle's", stage=S2)
def batching_workloads(ctx):
    bursts = workloads.bursty(ctx.rng, bursts=6, burst_size=7, period_s=120.0,
                              prompt_tokens=(4, 64), output_tokens=(1, 12))  # fmt: skip
    check_against_oracle(ctx, bursts, "bursts of seven, 3 slots", max_running=3)
    steady = workloads.poisson(ctx.rng, rate=0.02, n=100, prompt_tokens=(1, 128),
                               output_tokens=(1, 40), quantum_s=0.25)  # fmt: skip
    check_against_oracle(ctx, steady, "Poisson, 8 slots", max_running=8)
    check_against_oracle(ctx, steady, "Poisson, 1 slot", max_running=1, block_size=4)


# -- stage 3: chunked prefill ---------------------------------------------------------------

EXAMPLE_3 = [(0, 16, 4), (0, 40, 2), (3, 8, 2)]


@case("the stage 3 example: a budget of 32 tokens, chunks of at most 24", public=True, stage=S3)
def chunked_example(ctx):
    run = simulate(ctx, ctx.solution, workloads.replay(EXAMPLE_3), max_batch_tokens=32,
                   long_prefill_threshold=24)  # fmt: skip
    check_rules(run)
    check_steps(
        run,
        [
            (0, [(0, 16), (1, 16)], []),
            (16, [(1, 24), (2, 7)], [0]),
            (32.75, [(2, 1)], [0, 1]),
            (43, [], [0, 2]),
        ],
        "the stage 3 example",
    )
    check.equal(run.sim.trace.latencies("arrive", "first_token"), [16.0, 32.75, 40.0],
                label="times to first token")  # fmt: skip


@case("a prompt longer than the budget is split, and the next in line gets the rest",
      public=True, stage=S3)  # fmt: skip
def split_prompt(ctx):
    # Rejects waits_until_prompt_fits (the 100-token prompt never fits a budget of 48).
    requests = workloads.replay([(0, 100, 2), (0, 10, 2), (1, 6, 1)])
    run = simulate(ctx, ctx.solution, requests, max_batch_tokens=48)
    check_rules(run)
    check_steps(
        run,
        [
            (0, [(0, 48)], []),
            (20, [(0, 48)], []),
            (40, [(0, 4), (1, 10), (2, 6)], []),
            (53, [], [0, 1]),
        ],
        "a budget of 48",
    )


@case("decodes are served before prompts when the budget is tight", public=True, stage=S3)
def decodes_first(ctx):
    # Rejects prefill_starves_decodes: with prompts first, nothing decodes at t=11.
    requests = workloads.replay([(0, 4, 4), (0, 4, 4), (0, 4, 4), (2, 30, 1)])
    run = simulate(ctx, ctx.solution, requests, max_batch_tokens=12, max_running=4)
    check_rules(run)
    check_steps(
        run,
        [
            (0, [(0, 4), (1, 4), (2, 4)], []),
            (11, [(3, 9)], [0, 1, 2]),
            (24.25, [(3, 9)], [0, 1, 2]),
            (37.5, [(3, 9)], [0, 1, 2]),
            (50.75, [(3, 3)], []),
        ],
        "a budget of 12",
    )


@case("a long prompt does not stall the sequences that are generating", stage=S3)
def stall_bound(ctx):
    rows = [(0, 8, 60)] * 6 + [(30, 4000, 4), (31, 24, 4), (32, 24, 4), (33, 24, 4)]
    requests = workloads.replay(rows)
    for threshold in (None, 128):
        options = {} if threshold is None else {"long_prefill_threshold": threshold}
        run = simulate(ctx, ctx.solution, requests, max_batch_tokens=512, max_running=16,
                       **options)  # fmt: skip
        check_rules(run)
        times = {}
        for e in run.sim.trace.of("token"):
            times.setdefault(e["id"], []).append(e.time)
        gap = max(b - a for i in range(6) for a, b in pairwise(times[i]))
        if threshold is None:
            check.at_most("longest_token_gap", gap, 150.0, unit="ms")
        else:
            check.at_most("longest_token_gap_with_threshold", gap, 150.0, unit="ms")
            ttft = run.sim.trace.latencies("arrive", "first_token")
            check.at_most("short_prompt_ttft", max(ttft[7:]), 120.0, unit="ms")


@case("Poisson arrivals under a token budget: the steps equal the oracle's", stage=S3)
def chunked_workloads(ctx):
    requests = workloads.poisson(ctx.rng, rate=0.03, n=70, prompt_tokens=(1, 400),
                                 output_tokens=(1, 24), quantum_s=0.25)  # fmt: skip
    check_against_oracle(ctx, requests, "budget 64", max_batch_tokens=64, max_running=8)
    check_against_oracle(ctx, requests, "budget 256, threshold 48", max_batch_tokens=256,
                         max_running=12, long_prefill_threshold=48)  # fmt: skip
    check_against_oracle(ctx, requests, "budget 16, threshold 5", max_batch_tokens=16,
                         max_running=16, long_prefill_threshold=5, block_size=4)  # fmt: skip
    bursts = workloads.bursty(ctx.rng, bursts=5, burst_size=6, period_s=200.0,
                              prompt_tokens=(8, 200), output_tokens=(2, 16))  # fmt: skip
    check_against_oracle(ctx, bursts, "bursts, budget 96", max_batch_tokens=96, max_running=4)


# -- stage 4: priorities and preemption -----------------------------------------------------

EXAMPLE_4 = [(0, 8, 6, 1), (0, 6, 7, 0), (2, 7, 5, 2), (30, 4, 3, 0)]


@case("the stage 4 example: the least urgent sequence loses its memory", public=True, stage=S4)
def priority_example(ctx):
    # Rejects victim_ignores_priority (at 55.25 the newest sequence is request 3).
    run = simulate(ctx, ctx.solution, workloads.replay(EXAMPLE_4), n_blocks=7, block_size=4,
                   max_running=3)  # fmt: skip
    check_rules(run)
    check_steps(
        run,
        [
            (0, [(1, 6), (0, 8)], []),
            (11.5, [(2, 7)], [1, 0]),
            (23.25, [], [1, 0, 2]),
            (34.25, [], [1, 0]),
            (44.25, [(3, 4)], [1, 0]),
            (55.25, [], [1, 3]),
            (65.25, [], [1, 3]),
            (75.25, [(0, 13), (2, 9)], []),
            (88.75, [], [2]),
            (97.75, [], [2]),
        ],
        "the stage 4 example",
    )
    check.equal(run.preempts, [(34.25, 2), (55.25, 0)], label="preemptions (time, id)")
    check.cost(run.engine.counters, "recomputed_tokens", exactly=22, unit="tokens")


@case("equal priorities: the newest sequence is preempted and returns first", public=True,
      stage=S4)  # fmt: skip
def recompute_example(ctx):
    requests = workloads.replay([(0, 8, 6), (0, 8, 4), (4, 6, 5), (30, 4, 2)])
    run = simulate(ctx, ctx.solution, requests, n_blocks=7, block_size=4, max_running=3,
                   watermark_blocks=1)  # fmt: skip
    check_rules(run)
    check_steps(
        run,
        [
            (0, [(0, 8), (1, 8)], []),
            (12, [], [0, 1]),
            (22, [], [0, 1]),
            (32, [], [0, 1]),
            (42, [(2, 6), (3, 4)], [0]),
            (53.5, [], [0, 2]),
            (63.5, [(3, 5)], [2]),
            (73.75, [], [2]),
            (82.75, [], [2]),
        ],
        "one preemption",
    )
    check.equal(run.preempts, [(53.5, 3)], label="preemptions (time, id)")
    tokens = [(e.time, e["index"]) for e in run.sim.trace.of("token") if e["id"] == 3]
    check.equal(tokens, [(53.5, 0), (73.75, 1)], label="tokens of request 3 (time, index)")


@case("the queue is ordered by priority, then by arrival", public=True, stage=S4)
def queue_order(ctx):
    # Rejects queue_ignores_priority.
    requests = workloads.replay(
        [(0, 8, 3, 1), (1, 8, 2, 1), (2, 8, 2, 0), (3, 40, 2, -1), (4, 4, 2, 0)]
    )
    run = simulate(ctx, ctx.solution, requests, max_running=2)
    check_rules(run)
    check_steps(
        run,
        [
            (0, [(0, 8)], []),
            # Request 3 (priority -1) arrived last of the four waiting and is first in line.
            (10, [(3, 40)], [0]),
            (29, [], [3, 0]),
            # Requests 2 and 4 (priority 0) go before request 1, which arrived earlier.
            (39, [(2, 8), (4, 4)], []),
            (50, [], [2, 4]),
            (60, [(1, 8)], []),
            (70, [], [1]),
        ],
        "priorities in the queue",
    )


@case("the watermark keeps recomputation low on a pool that stays full", stage=S4)
def watermark_cost(ctx):
    # Rejects no_watermark: without headroom the same workload recomputes several
    # times as much.
    requests = workloads.poisson(ctx.rng, rate=0.012, n=120, prompt_tokens=(16, 160),
                                 output_tokens=(8, 96), priorities=(0, 1, 2), quantum_s=0.25)  # fmt: skip
    run = simulate(ctx, ctx.solution, requests, n_blocks=40, block_size=16, max_running=8,
                   max_batch_tokens=128, watermark_blocks=4)  # fmt: skip
    check_rules(run)
    check.cost(run.engine.counters, "recomputed_tokens", at_most=3000, unit="tokens")
    check.cost(run.engine.counters, "preemptions", at_most=30)


@case("memory pressure with priorities and a token budget: the trace equals the oracle's",
      stage=S4)  # fmt: skip
def pressure_workloads(ctx):
    bursts = workloads.bursty(ctx.rng, bursts=5, burst_size=6, period_s=250.0,
                              prompt_tokens=(4, 40), output_tokens=(4, 30), priorities=(0, 1))  # fmt: skip
    for watermark in (0, 2):
        check_against_oracle(ctx, bursts, f"bursts, watermark {watermark}", n_blocks=9,
                             block_size=8, max_running=6, watermark_blocks=watermark)  # fmt: skip
    steady = workloads.poisson(ctx.rng, rate=0.012, n=90, prompt_tokens=(16, 160),
                               output_tokens=(8, 96), priorities=(0, 1, 2), quantum_s=0.25)  # fmt: skip
    check_against_oracle(ctx, steady, "Poisson, three priorities", n_blocks=40, block_size=16,
                         max_running=8, watermark_blocks=4)  # fmt: skip
    check_against_oracle(ctx, steady, "Poisson, budget 96, threshold 40", n_blocks=30,
                         block_size=16, max_running=6, max_batch_tokens=96,
                         long_prefill_threshold=40, watermark_blocks=1)  # fmt: skip
    equal = workloads.poisson(ctx.rng, rate=0.015, n=60, prompt_tokens=(1, 20),
                              output_tokens=(1, 40), quantum_s=0.25)  # fmt: skip
    check_against_oracle(ctx, equal, "one priority, blocks of one token", n_blocks=64,
                         block_size=1, max_running=5)  # fmt: skip
    check_against_oracle(ctx, equal, "one priority, one slot", n_blocks=8, block_size=8,
                         max_running=1, watermark_blocks=3)  # fmt: skip


# -- stage 5: prefix sharing ------------------------------------------------------------------

SYSTEM = (1, 2, 3, 4, 5, 6, 7, 8)


def prompt_request(seq_id, arrival, tokens, output_tokens, priority=0):
    return SimRequest(seq_id, arrival, len(tokens), output_tokens, priority, tokens=tuple(tokens))


EXAMPLE_5 = [
    prompt_request(0, 0, (*SYSTEM, 11, 12, 13), 4),
    prompt_request(1, 0, (*SYSTEM, 21, 22), 3),
    prompt_request(2, 5, (*SYSTEM, 31, 32, 33, 34, 35), 2),
    prompt_request(3, 30, (1, 2, 3, 4, 9, 9, 9, 9, 9), 2),
    prompt_request(4, 40, (*SYSTEM, 41), 2),
]


@case("the stage 5 example: later requests share the blocks of a common prefix", public=True,
      stage=S5)  # fmt: skip
def sharing_example(ctx):
    run = simulate(ctx, ctx.solution, EXAMPLE_5, n_blocks=10, block_size=4, max_running=4,
                   prefix_sharing=True)  # fmt: skip
    check_rules(run)
    check_steps(
        run,
        [
            (0, [(0, 11), (1, 10)], []),
            (13.25, [(2, 5)], [0, 1]),
            (24.5, [], [0, 1, 2]),
            (35.5, [(3, 5)], [0]),
            (45.75, [(4, 5)], [3]),
            (56, [], [4]),
        ],
        "the stage 5 example",
    )
    check.equal(run.admits, [(0, 0, 0), (0, 1, 0), (13.25, 2, 8), (35.5, 3, 4), (45.75, 4, 4)],
                label="admissions (time, id, shared tokens)")  # fmt: skip
    check.equal(run.preempts, [], label="preemptions")
    check.cost(run.engine.counters, "prefill_tokens", exactly=36, unit="tokens")


@case("a prefix fork shares the blocks of the parent's first tokens", public=True, stage=S5)
def prefix_fork(ctx):
    # Rejects prefix_fork_no_refcount at the first audit after the fork.
    s = Script(ctx, n_blocks=8, block_size=4)
    s.append("a", 11, ([0, 1, 2], []))
    s.fork("a", "b", 8)  # two full blocks
    s.table("b", [0, 1])
    s.refs(1, 2)
    s.refs(2, 1)
    s.append("b", 3, ([3], []))  # b's last block was full: no copy
    s.free("a")  # blocks 0 and 1 stay: b holds them
    s.free_blocks(5)
    s.fork("b", "c", 6)  # 6 tokens: blocks 0 and 1, the second only half used by c
    s.append("c", 1, ([2], [(1, 2)]))  # c may not write into b's block 1
    s.table("c", [0, 2])
    check.raises(ValueError, s.alloc.fork, "b", "d", 12, label="fork of more tokens than exist")
    check.raises(ValueError, s.alloc.fork, "b", "d", 0, label="fork of zero tokens")
    s.free("b")
    s.free("c")
    s.free_blocks(8)


@case("only blocks that are completely prefilled are shared, and one token is always left",
      public=True, stage=S5)  # fmt: skip
def sharing_rules(ctx):
    # Rejects shares_unfilled_blocks (request 1) and shares_whole_prompt (request 2).
    long = tuple(range(100, 124))  # 24 tokens: six blocks of 4
    requests = [
        prompt_request(0, 0, long, 8),
        prompt_request(1, 1, (*long[:20], 7, 7), 2),  # admitted while 0 has 8 tokens prefilled
        prompt_request(2, 45, long[:16], 2),  # its whole prompt is cached: 12 are shared
    ]
    run = simulate(ctx, ctx.solution, requests, n_blocks=16, block_size=4, max_running=4,
                   max_batch_tokens=12, long_prefill_threshold=8, prefix_sharing=True)  # fmt: skip
    check_rules(run)
    check.equal(run.admits, [(0, 0, 0), (10, 1, 8), (52.5, 2, 12)],
                label="admissions (time, id, shared tokens)")  # fmt: skip
    check.cost(run.engine.counters, "prefill_tokens", exactly=24 + 14 + 4, unit="tokens")


@case("random allocator workloads with prefix forks agree with the oracle", stage=S5)
def allocator_random_prefix(ctx):
    def prop(rng):
        n_blocks = int(rng.choice([8, 24]))
        block_size = int(rng.choice([2, 4, 16]))
        random_ops(ctx, rng, n_blocks, block_size, 80, prefix_forks=True)

    check.forall(prop, trials=12)


def shared_workload(ctx, **options):
    defaults = {"rate": 0.02, "n": 120, "groups": 3, "prefix_tokens": (64, 160),
                "suffix_tokens": (4, 40), "output_tokens": (4, 32), "quantum_s": 0.25}  # fmt: skip
    return workloads.shared_prefixes(ctx.rng, **(defaults | options))


@case("sharing saves prefill and memory on prompts with common prefixes", stage=S5)
def sharing_cost(ctx):
    # 120 requests over three prefixes of 64 to 160 tokens. Without sharing the same
    # run prefills 18655 tokens and fills all 96 blocks.
    requests = shared_workload(ctx)
    run = simulate(ctx, ctx.solution, requests, n_blocks=96, block_size=16, max_running=8,
                   prefix_sharing=True)  # fmt: skip
    check_rules(run)
    check.cost(run.engine.counters, "prefill_tokens", at_most=6000, unit="tokens")
    check.at_most("peak_blocks", run.engine.peak_blocks, 64, unit="blocks")
    check.cost(run.engine.counters, "preemptions", exactly=0)
    off = simulate(ctx, ctx.solution, requests, n_blocks=96, block_size=16, max_running=8)
    check.that(not [a for a in off.admits if a[2]], "blocks were shared with prefix_sharing off",
               off.timeline())  # fmt: skip


@case("shared prompts: the trace equals the oracle's", stage=S5)
def sharing_workloads(ctx):
    requests = shared_workload(ctx)
    check_against_oracle(ctx, requests, "three prefixes", n_blocks=48, block_size=16,
                         max_running=8, watermark_blocks=2, prefix_sharing=True)  # fmt: skip
    check_against_oracle(ctx, requests, "three prefixes, blocks of 5", n_blocks=200,
                         block_size=5, max_running=6, prefix_sharing=True)  # fmt: skip


@case("shared prompts under memory pressure, priorities, and a budget", stage=S5)
def sharing_under_pressure(ctx):
    requests = shared_workload(ctx, n=100, groups=2, priorities=(0, 1, 2), rate=0.03)
    check_against_oracle(ctx, requests, "tight pool", n_blocks=30, block_size=16,
                         max_running=8, watermark_blocks=1, prefix_sharing=True)  # fmt: skip
    check_against_oracle(ctx, requests, "tight pool, budget 64", n_blocks=30, block_size=16,
                         max_running=8, max_batch_tokens=64, long_prefill_threshold=48,
                         watermark_blocks=2, prefix_sharing=True)  # fmt: skip
    bursts = workloads.bursty(ctx.rng, bursts=4, burst_size=6, period_s=300.0,
                              prompt_tokens=(8, 40), output_tokens=(4, 24))  # fmt: skip
    # Prompts without token ids take part in everything except sharing.
    check_against_oracle(ctx, bursts, "requests without token ids", n_blocks=12, block_size=8,
                         max_running=4, prefix_sharing=True)  # fmt: skip


@case("identical prompts and prompts that end at a block boundary", stage=S5)
def identical_prompts(ctx):
    # A prompt that is entirely cached still prefills its last block (shares_whole_prompt),
    # and two requests admitted in one step share nothing with each other.
    base = tuple(range(50, 82))  # 32 tokens: four blocks of 8
    rows = [(0, base, 6), (0, base, 3), (20, base, 2), (21, base[:16], 4), (22, base[:9], 1),
            (60, (*base, 1, 2, 3), 2), (61, base[:8], 3), (61, base[:24], 2)]  # fmt: skip
    requests = [prompt_request(i, t, tokens, out) for i, (t, tokens, out) in enumerate(rows)]
    check_against_oracle(ctx, requests, "one prompt, many requests", n_blocks=24, block_size=8,
                         max_running=6, prefix_sharing=True)  # fmt: skip
    check_against_oracle(ctx, requests, "one prompt, a budget of 20", n_blocks=24, block_size=8,
                         max_running=6, max_batch_tokens=20, prefix_sharing=True)  # fmt: skip
    check_against_oracle(ctx, requests, "one prompt, a tight pool", n_blocks=9, block_size=8,
                         max_running=4, watermark_blocks=1, prefix_sharing=True)  # fmt: skip
