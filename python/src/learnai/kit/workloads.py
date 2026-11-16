"""SimRequest and workloads: the traffic for serving simulations.

A ``SimRequest`` describes one inference request by the numbers a scheduler cares
about: when it arrives, how long its prompt is, and how many tokens it will generate.
The functions here build lists of them, sorted by arrival time:

    requests = workloads.poisson(ctx.rng, rate=20.0, n=200)        # steady random traffic
    requests = workloads.bursty(ctx.rng, bursts=5, burst_size=8, period_s=1.0)
    requests = workloads.replay([(0.0, 12, 4), (0.5, 100, 16)])    # hand-written
    requests = workloads.phases(ctx.rng, [(2.0, 5.0), (4.0, 40.0), (2.0, 5.0)])    # a load step
    requests = workloads.shared_prefixes(ctx.rng, rate=20.0, n=200, groups=4)      # with token ids

    workloads.drive(sim, requests, scheduler.submit)               # deliver them on time

Everything random comes from the generator passed in (``ctx.rng``), so a workload is
identical on every run.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SimRequest:
    """One request of a simulated workload.

    Attributes:
        id: unique within a workload; the generators number requests 0, 1, 2, ... in
            arrival order.
        arrival_s: virtual time at which the request reaches the server.
        prompt_tokens: prompt length, at least 1.
        output_tokens: how many tokens the model will generate before it stops, at
            least 1. A real server does not know this in advance; a problem says
            whether its scheduler may read it.
        priority: smaller is more urgent (the vLLM convention). Default 0.
        tenant: who sent it, for fairness problems. Default None.
        deadline_s: virtual time by which the request must have finished, an absolute
            time like ``arrival_s``; None for no deadline.
        tokens: the prompt's token ids, for problems that look at what a prompt
            *contains* (prefix caches, cache-aware routers); None when only the length
            matters. When given, ``prompt_tokens`` must equal ``len(tokens)``.
    """

    id: Hashable
    arrival_s: float
    prompt_tokens: int
    output_tokens: int
    priority: int = 0
    tenant: str | None = None
    deadline_s: float | None = None
    tokens: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.arrival_s < 0 or not math.isfinite(self.arrival_s):
            raise ValueError(f"arrival_s must be a finite time >= 0, got {self.arrival_s}")
        for name in ("prompt_tokens", "output_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an int >= 1, got {value!r}")
        if self.tokens is not None:
            tokens = tuple(self.tokens)
            if not all(isinstance(t, int) and not isinstance(t, bool) for t in tokens):
                raise ValueError("tokens must be a sequence of ints")
            if len(tokens) != self.prompt_tokens:
                raise ValueError(
                    f"tokens has {len(tokens)} ids but prompt_tokens is {self.prompt_tokens}"
                )
            object.__setattr__(self, "tokens", tokens)


#: A token count: a fixed int, or ``(low, high)`` for a uniform draw, both ends included.
TokenRange = int | tuple[int, int]


def poisson(
    rng: np.random.Generator,
    *,
    rate: float,
    n: int,
    prompt_tokens: TokenRange = (8, 64),
    output_tokens: TokenRange = (4, 32),
    priorities: Sequence[int] = (0,),
    tenants: Sequence[str | None] = (None,),
    deadline_after_s: float | None = None,
    start_s: float = 0.0,
    quantum_s: float | None = None,
) -> list[SimRequest]:
    """``n`` requests arriving as a Poisson process of ``rate`` requests per second.

    Gaps between arrivals are exponential with mean ``1 / rate``; the first request
    arrives one gap after ``start_s``. Token counts are uniform over their ranges, and
    ``priority`` and ``tenant`` are drawn uniformly from ``priorities`` and
    ``tenants``. ``deadline_after_s`` sets every deadline that long after arrival.

    ``quantum_s`` rounds every arrival down to a multiple of it. Use a value that is
    exact in binary (``2 ** -10``, about a millisecond) to get arrival times that
    are exactly representable and that sometimes coincide.
    """
    if rate <= 0:
        raise ValueError("rate must be positive")
    arrivals = start_s + np.cumsum(rng.exponential(1.0 / rate, size=n))
    return _build(rng, arrivals, prompt_tokens, output_tokens, priorities, tenants,
                  deadline_after_s, quantum_s)  # fmt: skip


def bursty(
    rng: np.random.Generator,
    *,
    bursts: int,
    burst_size: int,
    period_s: float,
    jitter_s: float = 0.0,
    prompt_tokens: TokenRange = (8, 64),
    output_tokens: TokenRange = (4, 32),
    priorities: Sequence[int] = (0,),
    tenants: Sequence[str | None] = (None,),
    deadline_after_s: float | None = None,
    start_s: float = 0.0,
    quantum_s: float | None = None,
) -> list[SimRequest]:
    """``bursts`` groups of ``burst_size`` requests, one group every ``period_s`` seconds.

    Burst ``b`` starts at ``start_s + b * period_s``; each of its requests arrives a
    uniform delay in ``[0, jitter_s)`` later. With ``jitter_s=0`` (the default) a whole
    burst arrives at exactly the same instant, in id order: the test for ties.
    """
    if bursts < 0 or burst_size < 0 or period_s < 0 or jitter_s < 0:
        raise ValueError("bursts, burst_size, period_s, and jitter_s must not be negative")
    starts = start_s + np.repeat(np.arange(bursts) * period_s, burst_size)
    arrivals = np.sort(starts + rng.uniform(0.0, 1.0, size=starts.size) * jitter_s, kind="stable")
    return _build(rng, arrivals, prompt_tokens, output_tokens, priorities, tenants,
                  deadline_after_s, quantum_s)  # fmt: skip


def phases(
    rng: np.random.Generator,
    phases: Sequence[tuple[float, float]],
    *,
    prompt_tokens: TokenRange = (8, 64),
    output_tokens: TokenRange = (4, 32),
    priorities: Sequence[int] = (0,),
    tenants: Sequence[str | None] = (None,),
    deadline_after_s: float | None = None,
    start_s: float = 0.0,
    quantum_s: float | None = None,
) -> list[SimRequest]:
    """A load that changes in steps: ``phases`` is a list of ``(duration_s, rate)``.

        workloads.phases(rng, [(2.0, 5.0), (4.0, 40.0), (2.0, 5.0)])     # quiet, busy, quiet

    During each phase requests arrive as a Poisson process of that phase's rate
    (requests per second; 0 for silence); phases follow one another from ``start_s``.
    The number of requests is whatever the draw gives (about the sum of ``duration *
    rate``). The other arguments are those of ``poisson``.
    """
    arrivals: list[float] = []
    begin = start_s
    for duration, rate in phases:
        if duration < 0 or rate < 0:
            raise ValueError("a phase is (duration_s >= 0, rate >= 0)")
        at = begin
        while rate > 0:
            at += float(rng.exponential(1.0 / rate))
            if at >= begin + duration:
                break
            arrivals.append(at)
        begin += duration
    return _build(rng, np.asarray(arrivals, dtype=np.float64), prompt_tokens, output_tokens,
                  priorities, tenants, deadline_after_s, quantum_s)  # fmt: skip


def token_sequences(
    rng: np.random.Generator,
    n: int,
    *,
    groups: int = 4,
    prefix_tokens: TokenRange = (32, 96),
    suffix_tokens: TokenRange = (4, 24),
    vocab_size: int = 64,
    skew: float = 0.0,
) -> list[tuple[int, ...]]:
    """``n`` token sequences that share prefixes, the way prompts share system prompts.

    ``groups`` prefixes are drawn first (lengths from ``prefix_tokens``, ids uniform in
    ``[0, vocab_size)``). Each sequence is one of the prefixes followed by a suffix of
    its own (length from ``suffix_tokens``). The group is uniform for ``skew=0``;
    with ``skew > 0`` group ``g`` is chosen with weight ``(g + 1) ** -skew``, so a
    few prefixes are hot. Two sequences of one group share exactly their prefix,
    except where their suffixes happen to start alike.
    """
    if groups < 1 or n < 0 or vocab_size < 2 or skew < 0:
        raise ValueError("need groups >= 1, n >= 0, vocab_size >= 2, skew >= 0")
    prefix_lengths = _draw(rng, prefix_tokens, groups)
    prefixes = [tuple(int(t) for t in rng.integers(0, vocab_size, int(k))) for k in prefix_lengths]
    weights = np.arange(1, groups + 1, dtype=np.float64) ** -skew
    chosen = rng.choice(groups, size=n, p=weights / weights.sum())
    suffix_lengths = _draw(rng, suffix_tokens, n)
    out = []
    for group, length in zip(chosen, suffix_lengths, strict=True):
        suffix = tuple(int(t) for t in rng.integers(0, vocab_size, int(length)))
        out.append(prefixes[int(group)] + suffix)
    return out


def shared_prefixes(
    rng: np.random.Generator,
    *,
    rate: float,
    n: int,
    groups: int = 4,
    prefix_tokens: TokenRange = (32, 96),
    suffix_tokens: TokenRange = (4, 24),
    vocab_size: int = 64,
    skew: float = 0.0,
    output_tokens: TokenRange = (4, 32),
    priorities: Sequence[int] = (0,),
    tenants: Sequence[str | None] = (None,),
    deadline_after_s: float | None = None,
    start_s: float = 0.0,
    quantum_s: float | None = None,
) -> list[SimRequest]:
    """Poisson traffic whose prompts are ``token_sequences``: every request has
    ``tokens``, and requests of one group share a prefix a cache can reuse."""
    if rate <= 0:
        raise ValueError("rate must be positive")
    arrivals = start_s + np.cumsum(rng.exponential(1.0 / rate, size=n))
    prompts = token_sequences(rng, n, groups=groups, prefix_tokens=prefix_tokens,
                              suffix_tokens=suffix_tokens, vocab_size=vocab_size, skew=skew)  # fmt: skip
    return _build(rng, arrivals, 1, output_tokens, priorities, tenants, deadline_after_s,
                  quantum_s, tokens=prompts)  # fmt: skip


def replay(rows: Iterable[SimRequest | Mapping[str, Any] | Sequence[Any]]) -> list[SimRequest]:
    """A hand-written workload, sorted by arrival time.

    Each row is a ``SimRequest``, a dict of its fields, or a tuple
    ``(arrival_s, prompt_tokens, output_tokens)`` optionally followed by ``priority``,
    ``tenant``, ``deadline_s``, and ``tokens``. Dicts and tuples get their position in ``rows`` as
    ``id`` unless a dict gives one. Rows with equal arrival times keep their order.
    Raises ``ValueError`` if two requests share an id.
    """
    requests = []
    for position, row in enumerate(rows):
        if isinstance(row, SimRequest):
            requests.append(row)
        elif isinstance(row, Mapping):
            requests.append(SimRequest(**{"id": position, **row}))
        else:
            requests.append(SimRequest(position, *row))
    ids = [request.id for request in requests]
    if len(set(ids)) != len(ids):
        raise ValueError("request ids must be unique")
    return sorted(requests, key=lambda request: request.arrival_s)


def drive(
    sim: Any,
    requests: Iterable[SimRequest],
    on_arrival: Callable[[SimRequest], Any],
    *,
    priority: int = 0,
) -> None:
    """Schedule every request's arrival on the simulated clock.

    At ``request.arrival_s`` an ``arrive`` event (fields ``id``, ``request``) is
    recorded in ``sim.trace`` and ``on_arrival(request)`` is called. If that returns a
    coroutine (``on_arrival`` is an ``async def``), it is started as a task with
    ``sim.spawn``. Requests arriving at the same instant are delivered in list order.
    Call this before ``sim.run()``; ``priority`` is the timer priority of the arrivals
    (section 5.6 of the authoring guide).
    """

    def arrive(request: SimRequest) -> None:
        sim.trace.record("arrive", id=request.id, request=request)
        result = on_arrival(request)
        if inspect.iscoroutine(result):
            sim.spawn(result)

    for request in requests:
        sim.call_at(request.arrival_s, arrive, request, priority=priority)


def _build(
    rng: np.random.Generator,
    arrivals: np.ndarray,
    prompt_tokens: TokenRange,
    output_tokens: TokenRange,
    priorities: Sequence[int],
    tenants: Sequence[str | None],
    deadline_after_s: float | None,
    quantum_s: float | None,
    *,
    tokens: Sequence[tuple[int, ...]] | None = None,
) -> list[SimRequest]:
    # Every column is always drawn, in this order, so changing one option does not
    # shift the random numbers the other columns get.
    n = len(arrivals)
    prompts = _draw(rng, prompt_tokens, n)
    outputs = _draw(rng, output_tokens, n)
    priority_index = rng.integers(0, len(priorities), size=n)
    tenant_index = rng.integers(0, len(tenants), size=n)
    requests = []
    for i in range(n):
        arrival = float(arrivals[i])
        if quantum_s is not None:
            arrival = math.floor(arrival / quantum_s) * quantum_s
        requests.append(
            SimRequest(
                id=i,
                arrival_s=arrival,
                prompt_tokens=int(prompts[i]) if tokens is None else len(tokens[i]),
                output_tokens=int(outputs[i]),
                priority=priorities[int(priority_index[i])],
                tenant=tenants[int(tenant_index[i])],
                deadline_s=None if deadline_after_s is None else arrival + deadline_after_s,
                tokens=None if tokens is None else tokens[i],
            )
        )
    return requests


def _draw(rng: np.random.Generator, spec: TokenRange, n: int) -> np.ndarray:
    low, high = (spec, spec) if isinstance(spec, int) else spec
    if not 1 <= low <= high:
        raise ValueError(f"a token range must satisfy 1 <= low <= high, got {spec!r}")
    return rng.integers(low, high + 1, size=n)
