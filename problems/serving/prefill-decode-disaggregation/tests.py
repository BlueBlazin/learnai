"""Tests for prefill-decode-disaggregation.

A `Coordinator` is put in front of the kit's `SimCluster`, whose clock counts
milliseconds here. The test reads what the coordinator did from the trace: which
replica every request was submitted to and when, and every KV transfer. Small
scenarios are compared with literal values, seeded workloads with the oracle's trace
and with rules computed here (every long prompt is prefilled in the prefill pool,
moved, and decoded in the decode pool; nothing is decoded before its transfer has
landed). On a mixed workload of chat requests and long documents, the attainment of
a time-to-first-token objective and of a time-per-output-token objective is compared
with a colocated cluster of the same size.
"""

import dataclasses

from learnai.judge import case, check
from learnai.kit import CostModel, SimCluster, SimRequest, workloads

# A step over P prompt tokens and D decoded sequences takes 8 + 0.25 * P + D ms.
COST = CostModel(base_s=8, prefill_s_per_token=0.25, decode_s_per_seq=1)
TRANSFER_MS_PER_TOKEN = 1 / 32
TPOT_SLO_MS = 16.0


def ttft_slo_ms(request):
    return 100.0 + 0.5 * request.prompt_tokens


class Run:
    def __init__(self, sim, cluster, requests, config):
        self.sim = sim
        self.cluster = cluster
        self.requests = requests
        self.config = config

    @property
    def routes(self):
        """[(time, request id, replica)] of every submit, in order."""
        return [(e.time, e["id"], e["replica"]) for e in self.sim.trace.of("route")]

    @property
    def transfers(self):
        """[(time, request id, src, dst)] of every transfer, in order."""
        return [(e.time, e["id"], e["src"], e["dst"]) for e in self.sim.trace.of("transfer")]

    def attainment(self):
        trace = self.sim.trace
        arrived = {e["id"]: e.time for e in trace.of("arrive")}
        first = {e["id"]: e.time for e in trace.of("first_token")}
        finish = {e["id"]: e.time for e in trace.of("finish")}
        ttft = tpot = 0
        for r in self.requests:
            if r.id in first and first[r.id] - arrived[r.id] <= ttft_slo_ms(r):
                ttft += 1
            done = r.id in finish and r.id in first
            if done and (finish[r.id] - first[r.id]) / (r.output_tokens - 1) <= TPOT_SLO_MS:
                tpot += 1
        return ttft / len(self.requests), tpot / len(self.requests)

    def timeline(self):
        prefill_ids, decode_ids, threshold = self.config
        lines = [
            (
                f"prefill pool {prefill_ids}, decode pool {decode_ids}, local_threshold={threshold}; "
                "a step takes 8 + 0.25 * P + D ms, a transfer 1/32 ms per prompt token"
            ),
            "timeline:",
        ]
        kinds = ("arrive", "route", "transfer", "transfer_done", "first_token", "finish",
                 "violation")  # fmt: skip
        shown = [e for e in self.sim.trace.events if e.kind in kinds]
        for e in shown[:90]:
            if e.kind == "arrive":
                r = e["request"]
                what = f"arrive    {r.id!r} (prompt {r.prompt_tokens}, output {r.output_tokens})"
            elif e.kind == "route":
                what = f"submit    {e['id']!r} to replica {e['replica']}"
            elif e.kind == "transfer":
                what = f"transfer  {e['id']!r} from {e['src']} to {e['dst']} ({e['duration']:g} ms)"
            elif e.kind == "transfer_done":
                what = f"landed    {e['id']!r} on {e['dst']}"
            elif e.kind == "violation":
                what = f"VIOLATION {e['code']}: {e['message']}"
            else:
                what = f"{e.kind:<9} {e['id']!r}"
            lines.append(f"  t={e.time:>9g}  {what}")
        if len(shown) > 90:
            lines.append(f"  ... {len(shown) - 90} more events")
        return "\n".join(lines) + "\n"


def simulate(ctx, module, requests, *, prefill_ids, decode_ids, threshold):
    sim = ctx.simclock()
    n = len(prefill_ids) + len(decode_ids)
    cluster = SimCluster(sim, COST, replicas=n, max_seqs=8,
                         transfer_s_per_token=TRANSFER_MS_PER_TOKEN)  # fmt: skip
    run = Run(sim, cluster, requests, (prefill_ids, decode_ids, threshold))
    coordinator = module.Coordinator(cluster, list(prefill_ids), list(decode_ids), threshold)
    workloads.drive(sim, requests, coordinator.serve)
    sim.run()
    if cluster.violations:
        first = cluster.violations[0]
        check.that(
            False,
            f"the coordinator misused the cluster ({first.code})",
            f"{first.code}: {first}\n" + run.timeline(),
        )
    return run


def colocated(ctx, requests, replicas):
    """The baseline: every replica does everything; requests go to the shortest queue."""
    sim = ctx.simclock()
    cluster = SimCluster(sim, COST, replicas=replicas, max_seqs=8)
    workloads.drive(sim, requests, cluster.submit)
    sim.run()
    return Run(sim, cluster, requests, None)


def check_rules(run):
    """What the statement promises about every request, whatever the placement."""
    detail = run.timeline()
    prefill_ids, decode_ids, threshold = run.config
    trace = run.sim.trace
    finished = {e["id"] for e in trace.of("finish")}
    missing = [r.id for r in run.requests if r.id not in finished]
    check.that(
        not missing, f"{len(missing)} of {len(run.requests)} requests never finished", detail
    )
    check.that(run.cluster.counters.failed == 0, "a request failed", detail)

    submits, moves, landed = {}, {}, {}
    for when, request_id, replica in run.routes:
        submits.setdefault(request_id, []).append((when, replica))
    for when, request_id, src, dst in run.transfers:
        moves.setdefault(request_id, []).append((when, src, dst))
    for e in trace.of("transfer_done"):
        landed[e["id"]] = e.time
    first = {e["id"]: e.time for e in trace.of("first_token")}
    for r in run.requests:
        where = f"request {r.id!r} (prompt {r.prompt_tokens})"
        mine = submits.get(r.id, [])
        if r.prompt_tokens < threshold:
            check.that(
                len(mine) == 1 and mine[0][1] in decode_ids and r.id not in moves,
                "a prompt below the threshold was not served entirely on one decode replica",
                detail + where + f": submits {mine}, transfers {moves.get(r.id, [])}\n",
            )
            continue
        check.that(
            len(mine) == 2 and mine[0][1] in prefill_ids and mine[1][1] in decode_ids,
            "a prompt at or above the threshold was not prefilled in the prefill pool and "
            "then decoded in the decode pool",
            detail + where + f": submits (time, replica) {mine}\n",
        )
        check.that(
            len(moves.get(r.id, [])) == 1 and moves[r.id][0][1:] == (mine[0][1], mine[1][1]),
            "a disaggregated request needs exactly one transfer, from the replica that "
            "prefilled it to the replica that decodes it",
            detail + where + f": submits {mine}, transfers {moves.get(r.id, [])}\n",
        )
        check.that(
            moves[r.id][0][0] >= first[r.id],
            "a KV cache was transferred before its prefill had finished",
            detail + where + "\n",
        )
        check.that(
            r.id in landed and mine[1][0] >= landed[r.id],
            "a request was handed to its decode replica before its KV cache had arrived",
            detail + where + f": submitted for decode at {mine[1][0]:g}\n",
        )
    # Decode replicas never run a long prefill: their decode steps are not stalled.
    by_id = {r.id: r for r in run.requests}
    for e in trace.of("step"):
        if e["engine"] in decode_ids:
            long = [i for i, _ in e["prefill"] if by_id[i].prompt_tokens >= threshold]
            check.that(
                not long,
                "a decode replica prefilled a prompt at or above the threshold",
                detail + f"replica {e['engine']} at t={e.time:g}: prefill={list(e['prefill'])}\n",
            )


def check_literal(run, routes, transfers):
    detail = run.timeline()
    detail += "expected submits (time, request, replica):\n"
    detail += "".join(f"  {r!r}\n" for r in routes)
    detail += "expected transfers (time, request, src, dst):\n"
    detail += "".join(f"  {t!r}\n" for t in transfers)
    got = run.routes
    wrong = next((i for i, (a, b) in enumerate(zip(got, routes, strict=False)) if a != b), None)
    check.that(
        wrong is None and len(got) == len(routes),
        "the submits are not the ones the policy prescribes"
        + (
            ""
            if wrong is None
            else f" (the first difference is submit {wrong}: got {got[wrong]!r})"
        ),
        detail,
    )
    check.that(
        run.transfers == transfers,
        "the transfers are not the ones the policy prescribes",
        detail,
    )


def check_against_oracle(ctx, requests, label, **config):
    want = simulate(ctx, ctx.oracle, requests, **config)
    got = simulate(ctx, ctx.solution, requests, **config)
    check_rules(got)
    mine, theirs = got.routes, want.routes
    wrong = next((i for i, (a, b) in enumerate(zip(mine, theirs, strict=False)) if a != b), None)
    if wrong is None and len(mine) != len(theirs):
        wrong = min(len(mine), len(theirs))
    if wrong is not None:
        a = mine[wrong] if wrong < len(mine) else None
        b = theirs[wrong] if wrong < len(theirs) else None
        check.that(
            False,
            f"{label}: submit {wrong} (time, request, replica) is not the one the policy "
            "prescribes",
            got.timeline() + f"submit {wrong}: got {a!r}\n          expected {b!r}\n",
        )
    check.that(
        got.transfers == want.transfers,
        f"{label}: the transfers differ from the ones the policy prescribes",
        got.timeline(),
    )
    return got


def mixed_workload(ctx, *, rate, n, long_fraction=0.1):
    """Chat requests (32 to 128 prompt tokens) with a share of long documents."""
    requests = workloads.poisson(
        ctx.rng, rate=rate, n=n, prompt_tokens=(32, 128), output_tokens=(32, 128)
    )
    is_long = ctx.rng.random(n) < long_fraction
    prompts = ctx.rng.integers(1500, 4001, size=n)
    outputs = ctx.rng.integers(16, 65, size=n)
    return [
        dataclasses.replace(r, prompt_tokens=int(prompts[i]), output_tokens=int(outputs[i]))
        if is_long[i]
        else r
        for i, r in enumerate(requests)
    ]


EXAMPLE = [(0, 16, 3), (0, 320, 3), (2, 64, 2), (90, 320, 2), (90, 320, 2)]


@case("the worked example: one prefill replica, two decode replicas, threshold 64",
      public=True)  # fmt: skip
def worked_example(ctx):
    run = simulate(ctx, ctx.solution, workloads.replay(EXAMPLE), prefill_ids=[0],
                   decode_ids=[1, 2], threshold=64)  # fmt: skip
    check_literal(
        run,
        [
            (0, 0, 1),  # 16 tokens: local, on the first idle decode replica
            (0, 1, 0),  # 320 tokens: prefill pool
            (2, 2, 0),  # 64 tokens is not below the threshold: prefill pool
            (90, 3, 0),
            (90, 4, 0),
            (98, 1, 1),  # request 1: first token at 88, transfer of 10 ms to replica 1
            (114, 2, 2),  # request 2: first token at 112; replica 1 is decoding request 1
            (290, 3, 1),  # requests 3 and 4 finish their prefill together at 280:
            (290, 4, 2),  # the transfer of 3 to replica 1 counts, so 4 goes to replica 2
        ],
        [(88, 1, 0, 1), (112, 2, 0, 2), (280, 3, 0, 1), (280, 4, 0, 2)],
    )
    check_rules(run)


@case("the threshold: a prompt of exactly local_threshold tokens is disaggregated",
      public=True)  # fmt: skip
def threshold_boundary(ctx):
    # Rejects threshold_inclusive, everything_disaggregated, nothing_disaggregated.
    requests = workloads.replay([(0, 99, 2), (1, 100, 2), (2, 101, 2), (3, 1, 2), (400, 100, 2)])
    for threshold in (100, 1, 102):
        run = simulate(ctx, ctx.solution, requests, prefill_ids=[0, 1], decode_ids=[2, 3],
                       threshold=threshold)  # fmt: skip
        check_rules(run)
        check_against_oracle(ctx, requests, f"local_threshold={threshold}", prefill_ids=[0, 1],
                             decode_ids=[2, 3], threshold=threshold)  # fmt: skip


@case("the decode replica is chosen when the prefill ends, counting transfers in flight",
      public=True)  # fmt: skip
def decode_choice(ctx):
    # Rejects decode_chosen_at_arrival and ignores_inflight_transfers.
    # Four long prompts arrive together and finish their prefill in one step; three
    # short requests occupy decode replica 1 only until t = 30.
    rows = [(0, 8, 3), (0, 8, 3), (0, 8, 3)] + [(0, 400, 4)] * 4 + [(500, 8, 2)]
    run = simulate(ctx, ctx.solution, workloads.replay(rows), prefill_ids=[0],
                   decode_ids=[1, 2], threshold=64)  # fmt: skip
    check_literal(
        run,
        [
            (0, 0, 1), (0, 1, 2), (0, 2, 1),  # short: alternate between the decode replicas
            (0, 3, 0), (0, 4, 0), (0, 5, 0), (0, 6, 0),
            # prefill of all four ends at 408; both decode replicas are idle by then
            (420.5, 3, 1), (420.5, 4, 2), (420.5, 5, 1), (420.5, 6, 2),
            (500, 7, 1),
        ],
        [(408, 3, 0, 1), (408, 4, 0, 2), (408, 5, 0, 1), (408, 6, 0, 2)],
    )  # fmt: skip
    check_rules(run)


@case("a mixed workload on four replicas: TTFT and TPOT attainment against colocated",
      public=True)  # fmt: skip
def beats_colocated(ctx):
    # Rejects everything_disaggregated (TTFT of short prompts), nothing_disaggregated
    # and colocated_everywhere (TPOT).
    requests = mixed_workload(ctx, rate=0.008, n=500)
    base_ttft, base_tpot = colocated(ctx, requests, 4).attainment()
    run = simulate(ctx, ctx.solution, requests, prefill_ids=[0], decode_ids=[1, 2, 3],
                   threshold=256)  # fmt: skip
    ttft, tpot = run.attainment()
    ctx.metric("colocated_ttft_attainment", round(100 * base_ttft, 1), unit="%")
    ctx.metric("colocated_tpot_attainment", round(100 * base_tpot, 1), unit="%")
    ctx.metric("ttft_attainment", round(100 * ttft, 1), unit="%", limit=round(100 * base_ttft, 1),
               higher_is_better=True)  # fmt: skip
    ctx.metric("tpot_attainment", round(100 * tpot, 1), unit="%",
               limit=round(100 * base_tpot + 5, 1), higher_is_better=True)  # fmt: skip
    check.that(
        ttft >= base_ttft,
        f"ttft_attainment: {100 * ttft:.1f}% is below the colocated cluster's "
        f"{100 * base_ttft:.1f}%",
    )
    check.that(
        tpot >= base_tpot + 0.05,
        f"tpot_attainment: {100 * tpot:.1f}% is not 5 points above the colocated cluster's "
        f"{100 * base_tpot:.1f}%",
    )
    check_rules(run)


@case("one prefill replica, three decode replicas: the trace equals the policy's")
def exact_one_prefill(ctx):
    requests = mixed_workload(ctx, rate=0.008, n=300)
    check_against_oracle(ctx, requests, "threshold 256", prefill_ids=[0], decode_ids=[1, 2, 3],
                         threshold=256)  # fmt: skip
    check_against_oracle(ctx, requests, "threshold 64", prefill_ids=[0], decode_ids=[1, 2, 3],
                         threshold=64)  # fmt: skip


@case("two prefill replicas of unequal backlog: the prefill replica is chosen by tokens")
def exact_two_prefill(ctx):
    # Rejects prefill_by_queue_len: one 4000-token prompt weighs more than three of 300.
    rows = [(0, 4000, 2), (1, 300, 2), (2, 300, 2), (3, 300, 2), (4, 300, 2), (5, 2000, 2)]
    run = simulate(ctx, ctx.solution, workloads.replay(rows), prefill_ids=[3, 1],
                   decode_ids=[0, 2], threshold=64)  # fmt: skip
    want = simulate(ctx, ctx.oracle, workloads.replay(rows), prefill_ids=[3, 1],
                    decode_ids=[0, 2], threshold=64)  # fmt: skip
    check_literal(run, want.routes, want.transfers)
    check_rules(run)
    requests = mixed_workload(ctx, rate=0.012, n=300, long_fraction=0.25)
    check_against_oracle(ctx, requests, "pools [0, 1] and [2, 3, 4]", prefill_ids=[0, 1],
                         decode_ids=[2, 3, 4], threshold=200)  # fmt: skip
    check_against_oracle(ctx, requests, "pools [4, 2] and [0, 3, 1]", prefill_ids=[4, 2],
                         decode_ids=[0, 3, 1], threshold=1600)  # fmt: skip


@case("bursts of long prompts: many transfers in flight at once")
def exact_bursts(ctx):
    # Rejects ignores_inflight_transfers and transfer_not_awaited.
    bursts = workloads.bursty(
        ctx.rng, bursts=6, burst_size=7, period_s=3000.0, prompt_tokens=(100, 900),
        output_tokens=(2, 40),
    )  # fmt: skip
    check_against_oracle(ctx, bursts, "bursts of seven", prefill_ids=[0, 1],
                         decode_ids=[2, 3, 4], threshold=300)  # fmt: skip
    same = [SimRequest(i, 10.0 * (i // 5), 512, 6) for i in range(20)]
    check_against_oracle(ctx, same, "equal prompts, five at a time", prefill_ids=[0],
                         decode_ids=[1, 2], threshold=512)  # fmt: skip
