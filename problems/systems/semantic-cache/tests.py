"""Tests for semantic-cache.

`play` runs a script of put and get calls against a new cache and compares every
`get` with a literal expectation (hand-made scripts) and with the oracle playing the
same script. The last public case drives the cache from a simulated clock: requests
arrive over time, a miss calls a model that takes 2 s, and the sequence of hits and
misses is compared with the oracle's; it also checks, independently of the oracle,
that no request received an answer generated for another question or namespace.

Embeddings in hand-made scripts are small integer vectors whose cosines (0.6, 0.8,
0.96, 1.0) are exact; random scripts keep similarities 1e-9 apart from each other and
from the threshold unless two entries share an embedding.
"""

import numpy as np

from learnai.judge import case, check

NS = ("small", "", None)
OTHER = ("large", "", None)


def fmt(op):
    if op[0] == "put":
        _, emb, response, now, ns = op
        return f"put({emb}, {response!r}, now={now:g}, namespace={ns!r})"
    _, emb, now, ns = op[:4]
    return f"get({emb}, now={now:g}, namespace={ns!r})"


def play(ctx, threshold, ttl_s, capacity, ops, want=None):
    """ops: ("put", emb, response, now, ns) or ("get", emb, now, ns). `want` lists the
    expected result of every get, in order; the oracle is always compared as well."""
    cache = ctx.solution.SemanticCache(threshold, ttl_s, capacity)
    oracle = ctx.oracle.SemanticCache(threshold, ttl_s, capacity)
    head = f"SemanticCache(threshold={threshold}, ttl_s={ttl_s}, capacity={capacity})\n"
    log, gets = [], 0
    for op in ops:
        if op[0] == "put":
            _, emb, response, now, ns = op
            oracle.put(list(emb), response, now, ns)
            result = cache.put(list(emb), response, now, ns)
            check.that(result is None, "put: must return None")
            log.append("  " + fmt(op))
            continue
        _, emb, now, ns = op
        expected = oracle.get(list(emb), now, ns)
        if want is not None:
            check.that(want[gets] == expected, "test bug: literal and oracle disagree", fmt(op))
        got = cache.get(list(emb), now, ns)
        log.append(f"  {fmt(op)} -> {got!r}")
        if got != expected:
            what = "a miss" if got is None else f"the response {got!r}"
            right = "a miss" if expected is None else f"the response {expected!r}"
            check.that(
                False,
                f"get number {gets + 1} of the script returned {what}, expected {right}",
                f"{head}" + "\n".join(log) + f"\nexpected: {expected!r}\n",
            )
        gets += 1
    return cache


def put(emb, response, now, ns=NS):
    return ("put", emb, response, now, ns)


def get(emb, now, ns=NS):
    return ("get", emb, now, ns)


EXAMPLE = [
    put([1, 0], "A", 0),
    put([3, 4], "B", 10),
    get([4, 3], 20),
    get([4, 3], 20, OTHER),
    get([1, 0], 59),
    get([1, 0], 60),
    put([0, 1], "C", 65),
    get([3, 4], 65),
    put([-1, 0], "D", 66),
    get([0, 1], 67),
]


# Rejects first_above_threshold, cross_namespace_hits, ttl_refreshed_on_hit,
# expires_one_tick_late, fifo_eviction, evicts_before_purging, strict_threshold.
@case("the worked example: ten calls on a cache of capacity 2", public=True)
def worked_example(ctx):
    play(ctx, 0.8, 60, 2, EXAMPLE, want=["B", None, "A", None, "B", "B"])


# Rejects caches_default_temperature, caches_tool_requests, namespace_without_system.
@case("namespace_of: model, system prompt, and max_tokens; None for sampled or tool requests",
      public=True)  # fmt: skip
def namespaces(ctx):
    user = {"role": "user", "content": "What is the capital of Peru?"}
    french = {"role": "system", "content": "Answer in French."}
    brief = {"role": "system", "content": "Be brief."}
    tool = {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
    base = {"model": "small", "temperature": 0, "messages": [french, user]}
    cases = [
        (base, ("small", "Answer in French.", None)),
        ({**base, "temperature": 0.7}, None),
        ({**base, "tools": [tool]}, None),
        ({**base, "tools": []}, ("small", "Answer in French.", None)),
        ({"model": "small", "messages": [french, user]}, None),
        ({**base, "temperature": 0.0, "max_tokens": 64}, ("small", "Answer in French.", 64)),
        ({**base, "model": "large", "top_p": 0.9, "seed": 7}, ("large", "Answer in French.", None)),
        ({**base, "messages": [user]}, ("small", "", None)),
        ({**base, "messages": [french, user, brief]}, ("small", "Answer in French.\nBe brief.", None)),
        ({**base, "messages": [user, {"role": "assistant", "content": "Lima."}, user]},
         ("small", "", None)),
        ({**base, "temperature": 1e-3}, None),
        ({**base, "temperature": 2, "tools": [tool]}, None),
    ]  # fmt: skip
    for request, want in cases:
        got = ctx.solution.namespace_of(dict(request))
        check.that(
            got == want and (want is None or isinstance(got, tuple)),
            f"namespace_of returned {got!r}, expected {want!r}",
            f"request: {request!r}\nexpected: {want!r}\n",
        )


# Rejects first_above_threshold, ties_to_oldest, dot_product (a long vector that is
# less similar must not win), strict_threshold.
@case("several matches: the most similar wins; equal embeddings: the newest wins", public=True)
def best_match(ctx):
    ops = [
        put([4, 3], "near", 0),
        put([30, 40], "far but long", 1),
        put([1, 0], "exact", 2),
        get([1, 0], 3),
        get([2, 0], 3),
        get([4, 3], 4),
        get([0, 5], 4),
        put([4, 3], "near, updated", 5),
        get([8, 6], 6),
        get([3, 4], 6),
        put([1, 0], "exact, updated", 7),
        put([1, 0], "exact, updated twice", 8),
        get([1, 0], 9),
    ]
    want = ["exact", "exact", "near", "far but long", "near, updated", "far but long",
            "exact, updated twice"]  # fmt: skip
    play(ctx, 0.8, 100, 10, ops, want=want)
    # A threshold of 1.0 still admits an exact repeat, and nothing else.
    play(ctx, 1.0, 100, 10, [put([1, 0], "x", 0), get([1, 0], 1), get([4, 3], 1)],
         want=["x", None])  # fmt: skip


# Rejects ttl_refreshed_on_hit, expires_one_tick_late, chooses_before_filtering.
@case("expiry: fixed when stored, not extended by hits; a stale better match does not hide "
      "a live one", public=True)  # fmt: skip
def expiry(ctx):
    ops = [
        put([1, 0], "old exact", 0),
        put([4, 3], "young near", 6),
        get([1, 0], 9.5),
        get([1, 0], 9.984375),
        get([1, 0], 10),
        get([1, 0], 15.984375),
        get([1, 0], 16),
        put([1, 0], "again", 20),
        get([1, 0], 29),
        get([1, 0], 29.5),
        get([1, 0], 30),
    ]
    want = ["old exact", "old exact", "young near", "young near", None, "again", "again", None]
    play(ctx, 0.8, 10, 5, ops, want=want)


# Rejects fifo_eviction, evicts_before_purging, capacity_per_namespace.
@case("eviction: least recently used, after expired entries are dropped, over all namespaces")
def eviction(ctx):
    a, b, c, d = [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
    # LRU: A is hit, so B goes when C arrives; then A goes when D arrives (C was hit).
    ops = [
        put(a, "A", 0), put(b, "B", 1), get(a, 2), put(c, "C", 3),
        get(a, 4), get(b, 4), get(c, 4), get(c, 5), get(a, 5), get(c, 6), put(d, "D", 7),
        get(a, 8), get(c, 8), get(d, 8),
    ]  # fmt: skip
    play(ctx, 0.9, 1000, 2, ops, want=["A", "A", None, "C", "C", "A", "C", None, "C", "D"])
    # A dead entry makes room: B is the least recently used and must survive.
    ops = [
        put(a, "A", 0), put(b, "B", 5), get(a, 6), put(c, "C", 11),
        get(a, 12), get(b, 12), get(c, 12),
    ]  # fmt: skip
    play(ctx, 0.9, 10, 2, ops, want=["A", None, "B", "C"])
    # One capacity for the whole cache, whatever the namespaces.
    ops = [
        put(a, "A", 0, NS), put(a, "A large", 1, OTHER), put(b, "B", 2, NS),
        get(a, 3, NS), get(a, 3, OTHER), get(b, 3, NS),
        put(c, "C large", 4, OTHER), get(a, 5, OTHER), get(b, 5, NS), get(c, 5, OTHER),
    ]  # fmt: skip
    play(ctx, 0.9, 1000, 2, ops, want=[None, "A large", "B", None, "B", "C large"])
    play(ctx, 0.9, 1000, 1, [put(a, "A", 0), put(b, "B", 1), get(a, 2), get(b, 2)],
         want=[None, "B"])  # fmt: skip


# Rejects cross_namespace_hits, dot_product, and strict_threshold.
@case("the same question under another model or system prompt is a miss")
def scoping(ctx):
    fr = ("small", "Answer in French.", None)
    en = ("small", "", None)
    capped = ("small", "", 16)
    ops = [
        put([2, 1, 2], "Lima", 0, en),
        get([2, 1, 2], 1, fr), get([2, 1, 2], 1, capped), get([2, 1, 2], 1, en),
        put([2, 1, 2], "Lima (fr)", 2, fr),
        get([4, 2, 4], 3, fr), get([4, 2, 4], 3, en), get([4, 2, 4], 3, ("large", "", None)),
        get([0.2, 0.1, 0.2], 4, en), get([20, 1, 2], 4, en),
    ]  # fmt: skip
    play(ctx, 0.95, 100, 8, ops, want=[None, None, "Lima", "Lima (fr)", "Lima", None, "Lima",
                                       None])  # fmt: skip
    # A similarity equal to the threshold is a match (0.6 exactly), in its namespace only.
    ops = [put([3, 4], "x", 0, fr), get([1, 0], 1, fr), get([1, 0], 1, en), get([0, 1], 1, fr)]
    play(ctx, 0.6, 100, 4, ops, want=["x", None, "x"])


def random_script(rng, n_ops, dim, threshold, ttl_s):
    """A random script whose similarities are well separated (see the module docstring)."""
    pool = [rng.normal(size=dim) for _ in range(8)]
    namespaces = [NS, OTHER, ("small", "Be brief.", 32)]
    stored, ops, now, deadlines = [], [], 0.0, []

    def draw():
        base = pool[int(rng.integers(len(pool)))]
        roll = rng.random()
        if roll < 0.35:
            return base  # an exact repeat: ties between entries with equal embeddings
        if roll < 0.5:
            return base * float(rng.uniform(0.5, 3.0))  # the same direction, another length
        return base + rng.normal(size=dim) * float(rng.choice([0.05, 0.3, 0.6]))

    def separated(vec):
        """No similarity to a different stored embedding is near the threshold or near
        another one, and (for a put) no stored embedding is parallel to this one."""
        if not stored:
            return True, True
        others = np.stack(stored)
        sims = others @ vec / (np.linalg.norm(others, axis=1) * np.linalg.norm(vec))
        sims = np.unique(sims[~(others == vec).all(axis=1)])
        gaps = np.diff(np.sort(np.append(sims, threshold)))
        return bool((gaps > 1e-6).all()), bool((np.abs(sims) < 1 - 1e-9).all())

    while len(ops) < n_ops:
        step = int(rng.integers(0, 80)) / 64
        due = [t for t in deadlines if now <= t <= now + 2]
        # Sometimes land exactly on the instant an entry expires.
        now = min(due) if due and rng.random() < 0.3 else now + step
        vec = draw()
        apart, not_parallel = separated(vec)
        if not apart:
            continue
        ns = namespaces[int(rng.integers(len(namespaces)))]
        emb = [float(x) for x in vec]
        if rng.random() < 0.45:
            # Two stored embeddings that differ must not be equally similar to a later
            # query; scaled copies would be, so only exact repeats are stored twice.
            if not not_parallel:
                continue
            stored.append(vec)
            deadlines.append(now + ttl_s)
            ops.append(put(emb, f"response {len(ops)}", now, ns))
        else:
            ops.append(get(emb, now, ns))
    return ops


@case("random scripts of 300 calls: three namespaces, repeats, expiry, and eviction")
def random_scripts(ctx):
    rng = ctx.rng
    configs = [(0.7, 8, 4), (0.9, 30, 6), (0.5, 3, 12), (0.99, 1000, 3), (0.8, 1000, 64),
               (0.6, 6, 64)]  # fmt: skip
    for threshold, ttl_s, capacity in configs:
        play(ctx, threshold, ttl_s, capacity, random_script(rng, 300, 6, threshold, ttl_s))


# -- the simulated workload ----------------------------------------------------------------

LLM_S = 2.0
TTL_S = 24.0
THRESHOLD = 0.9


def workload(rng, n):
    """n requests: (arrival, question, variant embedding, request dict)."""
    dim, n_questions = 24, 4
    bases = rng.normal(size=(n_questions, dim))
    bases /= np.linalg.norm(bases, axis=1, keepdims=True)
    systems = ["", "Answer in French."]
    out = []
    for i in range(n):
        q = int(rng.integers(n_questions))
        vec = bases[q] + rng.normal(size=dim) * 0.03  # a paraphrase of question q
        messages = [{"role": "user", "content": f"question {q}, wording {i}"}]
        system = systems[int(rng.integers(2))]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        request = {"model": ["small", "large"][int(rng.integers(2))], "messages": messages}
        roll = rng.random()
        if roll < 0.75:
            request["temperature"] = 0
        elif roll < 0.85:
            request["temperature"] = 0.7
        elif roll < 0.93:
            request["temperature"] = 0
            request["tools"] = [{"type": "function", "function": {"name": "lookup"}}]
        out.append((i * 0.5, q, vec, request))
    # The threshold separates paraphrases from other questions, so a correct cache
    # never returns an answer to a different question.
    vecs = np.stack([v / np.linalg.norm(v) for _, _, v, _ in out])
    sims = vecs @ vecs.T
    same = np.equal.outer([q for _, q, _, _ in out], [q for _, q, _, _ in out])
    check.that(sims[same].min() > THRESHOLD + 0.02, "test bug: paraphrases too far apart")
    check.that(sims[~same].max() < THRESHOLD - 0.02, "test bug: questions too close")
    return out


def serve(ctx, module, requests, capacity):
    """Drive a cache with the requests; returns [(outcome, answer)] per request."""
    sim = ctx.simclock()
    cache = module.SemanticCache(THRESHOLD, TTL_S, capacity)
    results = [None] * len(requests)

    def answer_of(q, request):
        system = "\n".join(m["content"] for m in request["messages"] if m["role"] == "system")
        return f"answer to question {q} by {request['model']} under {system!r}"

    async def client(i, q, vec, request):
        emb = [float(x) for x in vec]
        ns = module.namespace_of(dict(request))
        if ns is None:
            await sim.sleep(LLM_S)
            results[i] = ("bypass", answer_of(q, request))
            return
        hit = cache.get(emb, sim.now(), ns)
        if hit is not None:
            results[i] = ("hit", hit)
            return
        await sim.sleep(LLM_S)
        answer = answer_of(q, request)
        cache.put(emb, answer, sim.now(), ns)
        results[i] = ("miss", answer)

    for i, (when, q, vec, request) in enumerate(requests):
        sim.spawn_at(when, lambda i=i, q=q, vec=vec, request=request: client(i, q, vec, request))
    sim.run()
    return results


# Rejects every mutant that changes which entry is returned or when it dies, and the
# three namespace_of mutants (a cached answer for another system prompt, a sampled or
# tool request served from the cache).
@case("240 requests on a simulated clock: hits and misses equal the reference's, and no "
      "request gets an answer generated for another question or namespace", public=True)  # fmt: skip
def simulated_workload(ctx):
    requests = workload(ctx.rng, 240)
    got = serve(ctx, ctx.solution, requests, capacity=8)
    want = serve(ctx, ctx.oracle, requests, capacity=8)

    def describe(i):
        when, q, _, request = requests[i]
        return (
            f"request {i} at t={when:g} s: question {q}, model {request['model']!r}, "
            f"temperature {request.get('temperature', 'absent')}, "
            f"{'tools, ' if request.get('tools') else ''}"
            f"system messages {[m['content'] for m in request['messages'] if m['role'] == 'system']}\n"
        )

    for i, (outcome, answer) in enumerate(got):
        _, q, _, request = requests[i]
        cacheable = not request.get("tools") and request.get("temperature", 1.0) == 0
        check.that(
            cacheable or outcome == "bypass",
            f"request {i} must not use the cache (it is sampled or uses tools), but "
            "namespace_of gave it a namespace",
            describe(i),
        )
        system = "\n".join(m["content"] for m in request["messages"] if m["role"] == "system")
        right = f"answer to question {q} by {request['model']} under {system!r}"
        check.that(
            answer == right,
            f"request {i} was served a cached answer that was generated for a different "
            "question, model, or system prompt",
            f"{describe(i)}it received: {answer!r}\nexpected: {right!r}\n",
        )
    for i, ((outcome, _), (expected, _)) in enumerate(zip(got, want, strict=True)):
        check.that(
            outcome == expected,
            f"request {i} was a {outcome}, expected a {expected} (what the rules give at that "
            "moment)",
            f"{describe(i)}expected: {expected}\n"
            f"(the model takes {LLM_S:g} s, ttl_s={TTL_S:g}, threshold={THRESHOLD}, capacity=8)\n",
        )
    calls = sum(1 for outcome, _ in got if outcome != "hit")
    ctx.metric("model_calls", calls, unit="calls")
    ctx.metric("hit_rate", sum(1 for o, _ in got if o == "hit") / len(got))
