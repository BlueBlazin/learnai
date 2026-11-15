"""SimEngine: a simulated LLM inference engine for scheduling problems.

A real engine (vLLM, TGI, llama.cpp's server) runs a loop: every iteration it picks
some sequences, runs one forward pass over all of them, and each sequence that was
ready gets one more token. *What to put in each iteration* is the scheduling problem.
``SimEngine`` is that forward pass with the model taken out: it keeps the bookkeeping
and the clock, and leaves every decision to the scheduler under test.

    engine = SimEngine(sim, CostModel(base_s=0.004, prefill_s_per_token=0.0001,
                                      decode_s_per_seq=0.001), max_batch_tokens=512)
    engine.admit(request)                         # a SimRequest; costs no time
    result = await engine.step(prefill=[(request.id, 64)], decode=[3, 5])
    result.tokens                                 # ids that produced a token this step
    result.finished                               # ids that produced their last token

The life of a sequence
----------------------
1. ``admit(request)`` makes it *live*. It needs ``request.prompt_tokens`` tokens of
   prefill.
2. ``step(prefill=[(id, n)])`` processes ``n`` more prompt tokens. A prompt may be
   prefilled in one step or in chunks over several.
3. The step that completes the prefill produces the sequence's **first output token**
   (the forward pass over the last prompt token is what predicts it).
4. From the next step on, the sequence is listed in ``decode``; each such step produces
   one more token. A sequence cannot be decoded in the step that completes its prefill.
5. When it has produced ``request.output_tokens`` tokens it finishes: it stops being
   live and its memory is freed, at the end of that step, without a call.

Time
----
A step takes ``cost.step_s(prefill_tokens, decode_seqs)`` virtual seconds:

    base_s + prefill_s_per_token * (prompt tokens in the step) + decode_s_per_seq * len(decode)

The engine runs one step at a time. Tokens appear when the step *ends*. A step that
ends at instant ``t`` completes after every ordinary timer due at ``t`` (its timer has
priority ``STEP_DONE_PRIORITY``), so a request that arrives at exactly ``t`` is already
there when the scheduler wakes up to plan the next step.

Limits
------
* ``max_batch_tokens``: prompt tokens plus decoded sequences in one step.
* ``max_seqs``: live sequences.
* ``kv_blocks`` of ``block_tokens`` tokens each: the KV cache. A sequence holds
  ``ceil(kv_tokens / block_tokens)`` blocks, where ``kv_tokens`` grows by ``n`` for a
  prefill of ``n`` and by 1 for a decode, at the *start* of the step.

``preempt(id)`` frees a sequence's blocks but keeps it live and keeps the tokens it
has produced; before it can be decoded again its whole context (prompt plus produced
tokens) must be prefilled again, and the step that completes that prefill produces its
next token. ``release(id)`` drops a sequence altogether (a canceled request). Both may
be called while a step is running: the sequence's share of that step is then lost (it
gets no token from it), though the step takes as long as planned.

Violations
----------
Anything the engine cannot do raises ``ProtocolViolation`` with a ``code``.

``step`` checks, in this order, and raises for the first that fails:

    busy                 another step is running
    empty_step           nothing to prefill and nothing to decode
    duplicate_sequence   an id listed twice in the step (in either list, or in both)
    unknown_sequence     a prefill id that is not live (never admitted, finished, released)
    bad_prefill          a chunk of fewer than 1 or more than the remaining prefill tokens
                         (these two per prefill entry, in list order)
    unknown_sequence     a decode id that is not live
    prefill_incomplete   a decode id that still needs prefill
                         (these two per decode entry, in list order)
    budget_exceeded      more than max_batch_tokens in the step
    out_of_blocks        the step needs more KV blocks than the pool has (``OutOfBlocks``)

``admit`` checks ``duplicate_sequence`` (the id is already live), ``bad_request`` (a
token count that is not an int >= 1), then ``too_many_sequences``. ``preempt`` and
``release`` raise ``unknown_sequence`` for an id that is not live.

A rejected call costs no time and leaves the engine exactly as it was. Every
violation is also kept in ``engine.violations`` and recorded in the trace, so catching
the exception does not hide it from a test. (The read-only queries, such as
``engine.generated(id)``, also raise ``unknown_sequence`` for an id that is not live,
but asking is not a violation: nothing is recorded.)

Trace events (in ``sim.trace``)
-------------------------------
    admit        id                                   when admit() is called
    step         index, prefill, decode, tokens, duration     when a step starts
    first_token  id, step                             at the end of the step, before `token`
    token        id, index, step                      index counts the sequence's tokens from 0
    finish       id, tokens, step                     right after the sequence's last `token`
    step_done    index                                after the step's token events
    preempt      id, generated
    release      id, generated
    violation    code, message

An ``admit`` event also has ``cached`` or ``resumed`` when the sequence was admitted
with ``cached_tokens=`` or ``resume_from=`` (see ``admit``), and every event has
``engine`` when the engine was given a ``name``.

``prefill`` is a tuple of ``(id, n)`` pairs and ``decode`` a tuple of ids, as given.
At the end of a step, sequences are reported in the order of the ``prefill`` list,
then the ``decode`` list.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import Any

from whetstone.kit.counters import Counters

#: Timer priority of step completion. Arrivals and other timers default to 0, so at a
#: shared instant they run first.
STEP_DONE_PRIORITY = 10


@dataclass(frozen=True)
class CostModel:
    """How long a step takes, in virtual seconds.

    Sums of decimal fractions such as 0.001 are not exact in floating point. When a
    test compares times exactly, choose costs that are exact in binary (0.25, 0.125,
    ``2 ** -10``), or compare the solution's trace with the oracle's from the same
    simulation, which rounds identically.
    """

    base_s: float
    prefill_s_per_token: float
    decode_s_per_seq: float

    def __post_init__(self) -> None:
        if min(self.base_s, self.prefill_s_per_token, self.decode_s_per_seq) < 0:
            raise ValueError("costs must not be negative")

    def step_s(self, prefill_tokens: int, decode_seqs: int) -> float:
        """Duration of a step over ``prefill_tokens`` prompt tokens and ``decode_seqs`` decodes."""
        return (
            self.base_s
            + self.prefill_s_per_token * prefill_tokens
            + self.decode_s_per_seq * decode_seqs
        )


class ProtocolViolation(RuntimeError):
    """The scheduler asked the engine for something it cannot do. ``code`` says what."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class OutOfBlocks(ProtocolViolation):
    """A step needed more KV cache blocks than were free."""


@dataclass(frozen=True)
class StepResult:
    """What one step did.

    Attributes:
        index: the step's number, counting from 0.
        duration: its length in virtual seconds.
        tokens: ids of the sequences that produced a token, in report order.
        finished: ids of the sequences that produced their last token.
    """

    index: int
    duration: float
    tokens: tuple[Hashable, ...]
    finished: tuple[Hashable, ...]


@dataclass
class _Sequence:
    id: Hashable
    prompt_tokens: int
    output_tokens: int
    to_prefill: int  # tokens that must be prefilled before the next decode
    prefilled: int = 0
    generated: int = 0
    kv_tokens: int = 0
    preempted: bool = False  # has been preempted at least once: its prefill is recompute

    @property
    def remaining_prefill(self) -> int:
        return self.to_prefill - self.prefilled


class SimEngine:
    """A simulated inference engine. See the module docstring for the full contract.

    Args:
        sim: the simulated clock (``ctx.simclock()``).
        cost: the ``CostModel``.
        max_batch_tokens: per-step token budget; None for no limit.
        max_seqs: limit on live sequences; None for no limit.
        kv_blocks: size of the KV block pool; None for unlimited memory.
        block_tokens: tokens per KV block.
        name: for simulations with several engines on one clock: when given, every
            trace event of this engine carries ``engine=name``.

    Attributes:
        busy: whether a step is running.
        live: ids of the live sequences, in admission order.
        free_blocks: free KV blocks, or None when memory is unlimited.
        violations: every ``ProtocolViolation`` raised so far.
        busy_s: total virtual time spent in steps that have started.
        counters: ``steps``, ``prefill_tokens``, ``decode_tokens``, ``output_tokens``,
            ``recomputed_tokens`` (prefill tokens of sequences that had been
            preempted), ``preemptions``.
    """

    def __init__(
        self,
        sim: Any,
        cost: CostModel,
        *,
        max_batch_tokens: int | None = None,
        max_seqs: int | None = None,
        kv_blocks: int | None = None,
        block_tokens: int = 16,
        name: Hashable | None = None,
    ) -> None:
        if block_tokens < 1:
            raise ValueError("block_tokens must be at least 1")
        self.sim = sim
        self.name = name
        self.cost = cost
        self.max_batch_tokens = max_batch_tokens
        self.max_seqs = max_seqs
        self.kv_blocks = kv_blocks
        self.block_tokens = block_tokens
        self.busy = False
        self.busy_s = 0.0
        self.violations: list[ProtocolViolation] = []
        self.counters = Counters(
            "steps", "prefill_tokens", "decode_tokens", "output_tokens",
            "recomputed_tokens", "preemptions",
        )  # fmt: skip
        self._seqs: dict[Hashable, _Sequence] = {}  # insertion order is admission order
        self._in_flight: set[Hashable] = set()  # sequences the running step still works for

    # -- reading the engine's state ----------------------------------------------------

    @property
    def live(self) -> list[Hashable]:
        return list(self._seqs)

    @property
    def free_blocks(self) -> int | None:
        if self.kv_blocks is None:
            return None
        return self.kv_blocks - sum(self._blocks(seq.kv_tokens) for seq in self._seqs.values())

    def remaining_prefill(self, seq_id: Hashable) -> int:
        """Tokens still to prefill before the sequence can be decoded (0: ready)."""
        return self._get(seq_id, asking=True).remaining_prefill

    def generated(self, seq_id: Hashable) -> int:
        """Output tokens the sequence has produced so far."""
        return self._get(seq_id, asking=True).generated

    def kv_tokens(self, seq_id: Hashable) -> int:
        """Tokens the sequence holds in the KV cache."""
        return self._get(seq_id, asking=True).kv_tokens

    def blocks(self, seq_id: Hashable) -> int:
        """KV blocks the sequence holds: ``ceil(kv_tokens / block_tokens)``."""
        return self._blocks(self._get(seq_id, asking=True).kv_tokens)

    # -- admitting and removing sequences ------------------------------------------------

    def admit(self, request: Any, *, cached_tokens: int = 0, resume_from: int = 0) -> None:
        """Make ``request`` (anything with ``id``, ``prompt_tokens``, ``output_tokens``)
        a live sequence. Takes no virtual time and may be called while a step runs.

        Two options describe a sequence that does not start from nothing:

        ``cached_tokens=c`` (``0 <= c < prompt_tokens``): the first ``c`` prompt tokens
        are already in the KV cache (a prefix-cache hit). The sequence holds their
        blocks at once and only ``prompt_tokens - c`` tokens remain to prefill. At
        least one token is always prefilled: its forward pass produces the first
        output token.

        ``resume_from=g`` (``1 <= g < output_tokens``): the sequence was prefilled, and
        produced its first ``g`` tokens, on another engine, and its KV cache was
        brought here. It holds ``prompt_tokens + g - 1`` tokens of KV, needs no
        prefill, and its next decode produces token index ``g``.

        Either is refused with ``bad_request`` when out of range (or when both are
        given), and with ``out_of_blocks`` when the pool cannot hold the blocks.
        """
        if request.id in self._seqs:
            raise self._violation("duplicate_sequence", f"sequence {request.id!r} is already live")
        for name in ("prompt_tokens", "output_tokens"):
            value = getattr(request, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise self._violation("bad_request", f"{name} must be an int >= 1, got {value!r}")
        for name, value, limit in (
            ("cached_tokens", cached_tokens, request.prompt_tokens),
            ("resume_from", resume_from, request.output_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < limit:
                raise self._violation(
                    "bad_request", f"{name} must be an int in [0, {limit}), got {value!r}"
                )
        if cached_tokens and resume_from:
            raise self._violation("bad_request", "cached_tokens and resume_from exclude each other")
        if self.max_seqs is not None and len(self._seqs) >= self.max_seqs:
            raise self._violation(
                "too_many_sequences", f"the engine holds at most {self.max_seqs} live sequences"
            )
        held = request.prompt_tokens + resume_from - 1 if resume_from else cached_tokens
        if self.kv_blocks is not None and self._blocks(held) > self.free_blocks:
            raise self._violation(
                "out_of_blocks",
                f"sequence {request.id!r} arrives with {held} tokens of KV "
                f"({self._blocks(held)} blocks); {self.free_blocks} blocks are free",
            )
        seq = _Sequence(
            request.id, request.prompt_tokens, request.output_tokens,
            to_prefill=0 if resume_from else request.prompt_tokens - cached_tokens,
            generated=resume_from, kv_tokens=held,
        )  # fmt: skip
        self._seqs[request.id] = seq
        extra = {"cached": cached_tokens} if cached_tokens else {}
        if resume_from:
            extra["resumed"] = resume_from
        self._record("admit", id=request.id, **extra)

    def preempt(self, seq_id: Hashable) -> None:
        """Free the sequence's KV blocks; it stays live and keeps its produced tokens.

        Its whole context, ``prompt_tokens + generated(id)`` tokens, must be prefilled
        again; the step that completes that produces its next token.
        """
        seq = self._get(seq_id)
        seq.kv_tokens = seq.prefilled = 0
        seq.to_prefill = seq.prompt_tokens + seq.generated
        seq.preempted = True
        self._in_flight.discard(seq_id)
        self.counters.add("preemptions")
        self._record("preempt", id=seq_id, generated=seq.generated)

    def release(self, seq_id: Hashable) -> None:
        """Drop a live sequence (a canceled request) and free its KV blocks."""
        seq = self._get(seq_id)
        del self._seqs[seq_id]
        self._in_flight.discard(seq_id)
        self._record("release", id=seq_id, generated=seq.generated)

    # -- the forward pass ------------------------------------------------------------------

    async def step(
        self,
        prefill: Iterable[tuple[Hashable, int]] = (),
        decode: Iterable[Hashable] = (),
    ) -> StepResult:
        """Run one engine iteration and return when it ends, virtual seconds later.

        ``prefill`` lists ``(id, n_tokens)`` chunks of prompt to process; ``decode``
        lists ids that each produce one token. See the module docstring.
        """
        prefill = tuple((seq_id, n) for seq_id, n in prefill)
        decode = tuple(decode)
        self._check_step(prefill, decode)

        prefill_tokens = sum(n for _, n in prefill)
        duration = self.cost.step_s(prefill_tokens, len(decode))
        index = self.counters.steps
        for seq_id, n in prefill:
            seq = self._seqs[seq_id]
            seq.kv_tokens += n
            if seq.preempted:
                self.counters.add("recomputed_tokens", n)
        for seq_id in decode:
            self._seqs[seq_id].kv_tokens += 1
        self.counters.add("steps")
        self.counters.add("prefill_tokens", prefill_tokens)
        self.counters.add("decode_tokens", len(decode))
        self.busy = True
        self.busy_s += duration
        self._in_flight = {seq_id for seq_id, _ in prefill} | set(decode)
        self._record(
            "step", index=index, prefill=prefill, decode=decode,
            tokens=prefill_tokens + len(decode), duration=duration,
        )  # fmt: skip

        # The step ends when its timer fires, whether or not the caller is still
        # waiting: a forward pass that has started cannot be taken back.
        done = self.sim.loop.create_future()

        def complete() -> None:
            result = self._complete(index, duration, prefill, decode)
            if not done.done():
                done.set_result(result)

        self.sim.call_at(self.sim.now() + duration, complete, priority=STEP_DONE_PRIORITY)
        return await done

    def _check_step(
        self, prefill: tuple[tuple[Hashable, int], ...], decode: tuple[Hashable, ...]
    ) -> None:
        if self.busy:
            raise self._violation("busy", "the engine runs one step at a time")
        if not prefill and not decode:
            raise self._violation("empty_step", "a step needs at least one prefill or decode")
        ids = [seq_id for seq_id, _ in prefill] + list(decode)
        for seq_id in ids:
            if ids.count(seq_id) > 1:
                raise self._violation(
                    "duplicate_sequence", f"sequence {seq_id!r} is listed twice in one step"
                )
        for seq_id, n in prefill:
            remaining = self._get(seq_id).remaining_prefill
            if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= remaining:
                raise self._violation(
                    "bad_prefill",
                    f"cannot prefill {n!r} tokens of sequence {seq_id!r}: {remaining} remain",
                )
        for seq_id in decode:
            remaining = self._get(seq_id).remaining_prefill
            if remaining:
                raise self._violation(
                    "prefill_incomplete",
                    f"sequence {seq_id!r} cannot be decoded: {remaining} tokens of prefill remain",
                )
        tokens = sum(n for _, n in prefill) + len(decode)
        if self.max_batch_tokens is not None and tokens > self.max_batch_tokens:
            raise self._violation(
                "budget_exceeded",
                f"a step of {tokens} tokens exceeds max_batch_tokens {self.max_batch_tokens}",
            )
        if self.kv_blocks is not None:
            growth = dict(prefill) | dict.fromkeys(decode, 1)
            needed = sum(
                self._blocks(seq.kv_tokens + growth.get(seq.id, 0)) for seq in self._seqs.values()
            )
            if needed > self.kv_blocks:
                raise self._violation(
                    "out_of_blocks",
                    f"the step needs {needed} KV blocks in total, the pool has {self.kv_blocks}",
                )

    def _complete(
        self,
        index: int,
        duration: float,
        prefill: tuple[tuple[Hashable, int], ...],
        decode: tuple[Hashable, ...],
    ) -> StepResult:
        """The end of a step: advance every sequence it still works for, emit tokens."""
        tokens: list[Hashable] = []
        finished: list[Hashable] = []
        for seq_id, n in prefill:
            if seq_id in self._in_flight:
                seq = self._seqs[seq_id]
                seq.prefilled += n
                if not seq.remaining_prefill:
                    self._emit(seq, index, tokens, finished)
        for seq_id in decode:
            if seq_id in self._in_flight:
                self._emit(self._seqs[seq_id], index, tokens, finished)
        self._in_flight = set()
        self.busy = False
        self._record("step_done", index=index)
        return StepResult(index, duration, tuple(tokens), tuple(finished))

    def _emit(self, seq: _Sequence, step: int, tokens: list, finished: list) -> None:
        if seq.generated == 0:
            self._record("first_token", id=seq.id, step=step)
        self._record("token", id=seq.id, index=seq.generated, step=step)
        seq.generated += 1
        self.counters.add("output_tokens")
        tokens.append(seq.id)
        if seq.generated == seq.output_tokens:
            del self._seqs[seq.id]
            finished.append(seq.id)
            self._record("finish", id=seq.id, tokens=seq.generated, step=step)

    # -- helpers ---------------------------------------------------------------------------

    def _get(self, seq_id: Hashable, *, asking: bool = False) -> _Sequence:
        """The live sequence ``seq_id``. ``asking`` marks a read-only query, whose
        failure is raised but not recorded as a violation."""
        try:
            return self._seqs[seq_id]
        except KeyError, TypeError:
            message = f"sequence {seq_id!r} is not live"
            if asking:
                raise ProtocolViolation("unknown_sequence", message) from None
            raise self._violation("unknown_sequence", message) from None

    def _record(self, kind: str, **fields: Any) -> None:
        if self.name is not None:
            fields["engine"] = self.name
        self.sim.trace.record(kind, **fields)

    def _blocks(self, kv_tokens: int) -> int:
        return -(-kv_tokens // self.block_tokens)

    def _violation(self, code: str, message: str) -> ProtocolViolation:
        error = (OutOfBlocks if code == "out_of_blocks" else ProtocolViolation)(code, message)
        self.violations.append(error)
        self._record("violation", code=code, message=message)
        return error

    def __repr__(self) -> str:
        return f"SimEngine(live={len(self._seqs)}, busy={self.busy}, {self.counters!r})"
