"""Tests for metrics-endpoint.

Every case builds the app with `make_app(model, clock)`, serves it on a local socket,
sends a scripted workload to POST /generate, and then reads GET /metrics.

`Model` is the kit's ScriptedLM with two additions the problem needs: it advances a
fake clock by a scripted amount while it produces each piece, so every time to first
token is an exact, known number (no real time is measured anywhere), and it can fail
with RuntimeError at a chosen piece. `Expected` keeps the metric values the workload
must produce; `check_scrape` parses the exposition strictly, by the rules of the
Prometheus text format, and compares.

All scripted delays are binary fractions, so clock differences are exact and an
observation that is meant to sit on a bucket bound (0.25, 0.5, 1.0, 2.5) does.
"""

import math
import re
import threading
import time

import httpx

from learnai.judge import CheckFailure, case, check
from learnai.kit import ScriptedLM

BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
RUNNING = "vllm:num_requests_running"
PROMPT_TOKENS = "vllm:prompt_tokens_total"
GENERATION_TOKENS = "vllm:generation_tokens_total"
REQUESTS = "http_requests_total"
TTFT = "vllm:time_to_first_token_seconds"
TYPES = {
    RUNNING: "gauge",
    PROMPT_TOKENS: "counter",
    GENERATION_TOKENS: "counter",
    REQUESTS: "counter",
    TTFT: "histogram",
}

NAME = r"[a-zA-Z_:][a-zA-Z0-9_:]*"
SAMPLE = re.compile(rf"^({NAME})(?:\{{(.*)\}})?[ \t]+(\S+)(?:[ \t]+-?\d+)?$")
LABEL = re.compile(r'[ \t]*([a-zA-Z_][a-zA-Z0-9_]*)[ \t]*=[ \t]*"((?:[^"\\\n]|\\.)*)"[ \t]*(?:,|$)')
VALUE = re.compile(r"^[+-]?(?:inf|infinity|nan|(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)$", re.IGNORECASE)


# -- the model and the clock ---------------------------------------------------------------


class Clock:
    """The `clock` handed to make_app: time passes only when the model says so."""

    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class Plan:
    """What the model does for one prompt.

    pieces: the script. delays: seconds the clock advances while each piece is being
    produced (missing entries are 0). fail_at: the model raises RuntimeError instead of
    producing piece number fail_at (0-based), after `fail_delay` more seconds.
    """

    def __init__(self, pieces, delays=(), fail_at=None, fail_delay=0.0):
        self.pieces, self.delays = list(pieces), list(delays)
        self.fail_at, self.fail_delay = fail_at, fail_delay

    def delay(self, i):
        return self.delays[i] if i < len(self.delays) else 0.0

    def outcome(self, max_tokens):
        """(status, pieces produced, time to first token or None) for a valid request."""
        limit = min(len(self.pieces), max_tokens)
        failed = self.fail_at is not None and self.fail_at <= limit
        produced = self.fail_at if failed else limit
        return (500 if failed else 200), produced, (self.delay(0) if produced else None)


class Model:
    """The `lm` handed to make_app: a ScriptedLM that keeps time and can fail."""

    def __init__(self, clock, plans, *, gated=False):
        self.clock, self.plans = clock, plans
        self.inner = ScriptedLM(lambda prompt: self.plans[prompt].pieces, gated=gated,
                                gate_timeout_s=3.0)  # fmt: skip

    def count_tokens(self, text):
        return self.inner.count_tokens(text)

    def stream(self, prompt, max_tokens=None):
        return _Stream(self, self.plans[prompt], self.inner.stream(prompt, max_tokens))


class _Stream:
    def __init__(self, model, plan, inner):
        self.model, self.plan, self.inner, self.produced = model, plan, inner, 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.plan.fail_at == self.produced:
            self.model.clock.advance(self.plan.fail_delay)
            await self.inner.aclose()
            raise RuntimeError("the model failed")
        piece = await self.inner.__anext__()
        self.model.clock.advance(self.plan.delay(self.produced))
        self.produced += 1
        return piece

    async def aclose(self):
        await self.inner.aclose()


# -- what the metrics must say ---------------------------------------------------------------


class Expected:
    def __init__(self):
        self.running = 0
        self.prompt_tokens = 0
        self.generation_tokens = 0
        self.requests = {"200": 0, "400": 0, "500": 0}
        self.observations = []
        self.history = []  # one line per request, for failure details

    def record(self, line):
        self.history.append(line)

    def workload(self):
        shown = self.history[-25:]
        more = f"  ... {len(self.history) - 25} earlier requests\n" if len(self.history) > 25 else ""
        return "requests sent so far:\n" + more + "".join(f"  {line}\n" for line in shown)


def serve(ctx, model, clock):
    return ctx.serve(ctx.solution.make_app(model, clock))


def send(client, method, path, label, **kwargs):
    try:
        return client.request(method, path, **kwargs)
    except httpx.TimeoutException:
        raise CheckFailure(f"{label}: timed out waiting for the response") from None
    except httpx.HTTPError as exc:
        raise CheckFailure(
            f"{label}: the connection broke: {type(exc).__name__}", f"{exc}\n"
        ) from None


def generate(client, model, expected, prompt, max_tokens=None):
    """POST a valid request, check the response, and update `expected`."""
    body = {"prompt": prompt}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    plan = model.plans[prompt]
    status, produced, ttft = plan.outcome(16 if max_tokens is None else max_tokens)
    label = f"POST /generate {body!r}"
    response = send(client, "POST", "/generate", label, json=body)
    expected.record(f"{label} -> {status}" + (f", first token after {ttft} s" if ttft else ""))
    expected.requests[str(status)] += 1
    expected.prompt_tokens += len(prompt.split())
    expected.generation_tokens += produced
    if ttft is not None:
        expected.observations.append(ttft)
    detail = f"status: {response.status_code}\nbody: {response.text[:300]!r}\n"
    check.that(
        response.status_code == status,
        f"{label}: got status {response.status_code}, expected {status}",
        detail,
    )
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if status == 200:
        want = {
            "text": "".join(plan.pieces[:produced]),
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": produced,
        }
        check.that(
            payload == want,
            f"{label}: the 200 response is not the JSON object with text, prompt_tokens, "
            "and completion_tokens",
            detail + f"expected: {want!r}\n",
        )
    else:
        check_error_body(payload, label, detail)


def bad_request(client, expected, body):
    label = f"POST /generate with body {body!r}"
    response = send(client, "POST", "/generate", label, content=body,
                    headers={"content-type": "application/json"})  # fmt: skip
    expected.record(f"{label} -> 400")
    expected.requests["400"] += 1
    detail = f"status: {response.status_code}\nbody: {response.text[:300]!r}\n"
    check.that(
        response.status_code == 400,
        f"{label}: got status {response.status_code}, expected 400",
        detail,
    )
    try:
        payload = response.json()
    except ValueError:
        payload = None
    check_error_body(payload, label, detail)


def check_error_body(payload, label, detail):
    check.that(
        isinstance(payload, dict) and isinstance(payload.get("error"), str) and payload["error"],
        f'{label}: the error body is not a JSON object {{"error": "<message>"}}',
        detail,
    )


# -- the exposition format --------------------------------------------------------------------


def parse_exposition(text):
    """Parse the text format strictly. Returns (types, samples).

    types: metric name -> type, from the TYPE lines.
    samples: [(name, labels dict, value)] in order of appearance.
    Any violation of the format fails the case.
    """

    def fail(message, line=None):
        shown = f"offending line: {line!r}\n" if line is not None else ""
        raise CheckFailure(f"GET /metrics: {message}", shown + f"the exposition:\n{text}")

    if not text.endswith("\n"):
        fail("the last line must end with a line feed")
    types, helps, samples, seen = {}, set(), [], set()
    order = []  # metric family of every sample, to check grouping

    def family_of(name):
        for suffix in ("_bucket", "_sum", "_count"):
            base = name.removesuffix(suffix)
            if base != name and types.get(base) == "histogram":
                return base
        return name

    for line in text.split("\n")[:-1]:
        if "\r" in line:
            fail("lines must end with a line feed only, not CR LF", line)
        if not line.strip():
            continue
        if line.startswith("#"):
            tokens = line[1:].split(None, 3)
            if tokens[:1] == ["TYPE"]:
                if len(tokens) != 3 or not re.fullmatch(NAME, tokens[1]):
                    fail("a TYPE line must be `# TYPE <metric name> <type>`", line)
                name, kind = tokens[1], tokens[2]
                if kind not in ("counter", "gauge", "histogram", "summary", "untyped"):
                    fail("unknown metric type in a TYPE line", line)
                if name in types:
                    fail("a metric has more than one TYPE line", line)
                if name in order:
                    fail("a TYPE line must come before the first sample of its metric", line)
                types[name] = kind
            elif tokens[:1] == ["HELP"]:
                if len(tokens) < 2 or not re.fullmatch(NAME, tokens[1]):
                    fail("a HELP line must be `# HELP <metric name> <text>`", line)
                if tokens[1] in helps:
                    fail("a metric has more than one HELP line", line)
                helps.add(tokens[1])
            continue
        match = SAMPLE.match(line)
        if not match:
            fail('a sample line must be `<name> <value>` or `<name>{<label>="<value>"} <value>`',
                 line)  # fmt: skip
        name, label_text, value_text = match.groups()
        labels = {}
        if label_text:
            position = 0
            while position < len(label_text):
                pair = LABEL.match(label_text, position)
                if not pair or pair.group(1) in labels:
                    fail('labels must be `name="value"` pairs separated by commas', line)
                labels[pair.group(1)] = pair.group(2)
                position = pair.end()
        if not VALUE.match(value_text):
            fail("a sample value is not a number", line)
        key = (name, tuple(sorted(labels.items())))
        if key in seen:
            fail("the same metric name and labels appear on two lines", line)
        seen.add(key)
        family = family_of(name)
        if family in order and order[-1] != family:
            fail("all lines of one metric must form a single group", line)
        order.append(family)
        samples.append((name, labels, float(value_text)))
    return types, samples


def check_scrape(client, expected, label):
    """GET /metrics and compare every required metric with `expected`."""
    response = send(client, "GET", "/metrics", "GET /metrics")
    check.equal(response.status_code, 200, label="status of GET /metrics")
    content_type = response.headers.get("content-type", "")
    parts = [part.strip().lower() for part in content_type.split(";")]
    check.that(
        parts[0] == "text/plain" and all(p == "version=0.0.4" for p in parts if "version" in p),
        "GET /metrics: the content type must be text/plain; version=0.0.4",
        f"content-type: {content_type!r}\n",
    )
    try:
        text = response.content.decode("utf-8")
    except UnicodeDecodeError:
        raise CheckFailure("GET /metrics: the body is not valid UTF-8") from None
    types, samples = parse_exposition(text)
    context = f"{label}\n{expected.workload()}the exposition:\n{text}"

    for name, kind in TYPES.items():
        check.that(
            types.get(name) == kind,
            f"GET /metrics: `# TYPE {name} {kind}` is missing or has another type",
            context,
        )

    def series(name):
        return [(labels, value) for sample, labels, value in samples if sample == name]

    def single(name, want, what):
        found = series(name)
        check.that(
            len(found) == 1 and found[0][0] == {},
            f"GET /metrics: {name} must be exactly one sample without labels",
            context,
        )
        check.that(
            found[0][1] == want,
            f"GET /metrics: {name} has the wrong value ({what})",
            context + f"expected {name} = {want}\n",
        )

    single(RUNNING, expected.running, "requests being generated right now")
    single(PROMPT_TOKENS, expected.prompt_tokens, "prompt tokens of all valid requests")
    single(GENERATION_TOKENS, expected.generation_tokens, "pieces received from the model")

    by_status = {}
    for labels, value in series(REQUESTS):
        check.that(
            set(labels) == {"status"},
            f"GET /metrics: every {REQUESTS} sample must have exactly the label `status`",
            context,
        )
        by_status[labels["status"]] = value
    for status, want in expected.requests.items():
        check.that(
            status in by_status,
            f'GET /metrics: {REQUESTS}{{status="{status}"}} is missing '
            "(all three statuses must be present, also at 0)",
            context,
        )
        check.that(
            by_status[status] == want,
            f'GET /metrics: {REQUESTS}{{status="{status}"}} has the wrong value',
            context + f"expected {want}\n",
        )

    buckets = []
    for labels, value in series(f"{TTFT}_bucket"):
        try:
            bound = float(labels["le"]) if set(labels) == {"le"} else None
        except ValueError:
            bound = None
        check.that(
            bound is not None and not math.isnan(bound),
            f"GET /metrics: every {TTFT}_bucket sample must have exactly the label `le`, "
            "a number or +Inf",
            context,
        )
        buckets.append((bound, value))
    bounds = [bound for bound, _ in buckets]
    check.that(
        bool(bounds) and bounds[-1] == math.inf,
        'GET /metrics: the histogram must end with the bucket le="+Inf"',
        context,
    )
    check.that(
        bounds == [*BUCKETS, math.inf],
        "GET /metrics: the histogram's buckets must be exactly TTFT_BUCKETS and +Inf, "
        "in increasing order",
        context,
    )
    observations = expected.observations
    want_counts = [sum(v <= bound for v in observations) for bound in bounds]
    check.that(
        [value for _, value in buckets] == want_counts,
        "GET /metrics: the histogram's bucket counts are "
        f"{', '.join(f'{value:g}' for _, value in buckets)}; expected "
        f"{', '.join(map(str, want_counts))} (each bucket counts the observations less than or "
        "equal to its bound)",
        context + f"observed times to first token: {observations}\n"
        f"expected bucket counts: {want_counts}\n",
    )
    for suffix, want in (("_count", len(observations)), ("_sum", sum(observations))):
        found = series(TTFT + suffix)
        check.that(
            len(found) == 1 and found[0][0] == {},
            f"GET /metrics: {TTFT}{suffix} must be exactly one sample without labels",
            context,
        )
        check.that(
            abs(found[0][1] - want) <= 1e-9,
            f"GET /metrics: {TTFT}{suffix} is {found[0][1]!r}, expected {want!r}",
            context + f"observed times to first token: {observations}\nexpected {want}\n",
        )
    return text


# -- cases ------------------------------------------------------------------------------------

EXAMPLE_PLANS = {
    "the quick brown fox": Plan(["Hello", ",", " world", "!"], [0.25, 0.5, 0.5, 0.5]),
    "hi": Plan(["Hey"], [0.03125]),
    "tell me a story": Plan(["Once", " upon", " a", " time"], [1.5, 0.125], fail_at=2),
}


@case("the worked example: two successes, a bad request, and a model failure", public=True)
def worked_example(ctx):
    # Rejects non_cumulative_buckets, le_exclusive (0.25 s is on a bound),
    # ttft_is_total_time, tokens_only_on_success, bad_requests_not_counted, real_clock.
    clock = Clock()
    model = Model(clock, EXAMPLE_PLANS)
    expected = Expected()
    with serve(ctx, model, clock) as client:
        generate(client, model, expected, "the quick brown fox", 3)
        generate(client, model, expected, "hi")
        bad_request(client, expected, b'{"prompt": 7}')
        generate(client, model, expected, "tell me a story")
        check_scrape(client, expected, "after the four requests of the worked example")
    # The statement's numbers, literally.
    assert (expected.prompt_tokens, expected.generation_tokens) == (9, 6)
    assert expected.requests == {"200": 2, "400": 1, "500": 1}
    assert expected.observations == [0.25, 0.03125, 1.5]


@case("a fresh server: every metric is present and zero, in the text format", public=True)
def fresh_server(ctx):
    # Rejects lazy_status_series, missing_inf_bucket, no_type_lines, metrics_as_json.
    clock = Clock()
    model = Model(clock, EXAMPLE_PLANS)
    with serve(ctx, model, clock) as client:
        first = check_scrape(client, Expected(), "before any request")
        second = check_scrape(client, Expected(), "a second scrape before any request")
        check.that(first == second, "two scrapes of an idle server differ")
        response = send(client, "GET", "/nope", "GET /nope")
        check.equal(response.status_code, 404, label="status of an unknown path")
        check_scrape(client, Expected(), "after a request to an unknown path")


@case("histogram buckets: observations on, between, and beyond the bounds", public=True)
def histogram_buckets(ctx):
    # Rejects le_exclusive, non_cumulative_buckets, missing_inf_bucket.
    ttfts = [0.25, 0.5, 1.0, 2.5, 0.046875, 0.0625, 0.09375, 0.125, 0.75, 2.515625, 60.0, 0.0]
    clock = Clock()
    plans = {f"p{i}": Plan(["a", "b"], [ttft, 0.5]) for i, ttft in enumerate(ttfts)}
    model = Model(clock, plans)
    expected = Expected()
    with serve(ctx, model, clock) as client:
        for i in range(len(ttfts)):
            generate(client, model, expected, f"p{i}")
        check_scrape(client, expected, "after twelve requests")


def wait_for(condition, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.002)
    return True


def scrape_until_idle(client, expected, label):
    """Scrape, allowing a moment for handlers that finish just after their response."""
    deadline = time.monotonic() + 0.5
    while True:
        try:
            return check_scrape(client, expected, label)
        except CheckFailure:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.02)


@case("the gauge counts requests in flight and returns to 0, also after a failure", public=True)
def running_gauge(ctx):
    # Rejects gauge_leak_on_error.
    clock = Clock()
    plans = {
        "one two": Plan(["x", "y"]),
        "three": Plan(["z"]),
        "broken at once": Plan(["never"], fail_at=0),
        "broken later": Plan(["a", "b", "c"], fail_at=2),
    }
    model = Model(clock, plans, gated=True)
    expected = Expected()
    with serve(ctx, model, clock) as client:
        threads = [
            threading.Thread(target=generate, args=(client, model, Expected(), prompt), daemon=True)
            for prompt in ("one two", "three")
        ]
        for thread in threads:
            thread.start()
        started = wait_for(lambda: model.inner.active == 2)
        check.that(started, "two concurrent requests did not both reach the model within 2 s")
        # Both generations are blocked on the model. Only the gauge is checked here.
        text = send(client, "GET", "/metrics", "GET /metrics during two generations").text
        _, samples = parse_exposition(text)
        running = [value for name, _, value in samples if name == RUNNING]
        model.inner.release()
        for thread in threads:
            thread.join(5.0)
        check.that(
            running == [2.0],
            f"{RUNNING} must be 2 while two requests are being generated",
            f"samples of {RUNNING}: {running}\n",
        )
        check.that(not model.inner.stalled, "a generation stalled: the model was not read")
        expected.record("POST /generate 'one two' and 'three', concurrently -> 200, 200")
        expected.requests["200"] += 2
        expected.prompt_tokens += 3
        expected.generation_tokens += 3
        expected.observations += [0.0, 0.0]
        scrape_until_idle(client, expected, "after both generations completed")
        generate(client, model, expected, "broken at once")
        scrape_until_idle(client, expected, "after a generation that failed immediately")
        generate(client, model, expected, "broken later")
        generate(client, model, expected, "three")
        scrape_until_idle(client, expected, "after a generation that failed at its third piece")


@case("failures, empty generations, and max_tokens: what each adds to the counters")
def counters(ctx):
    # Rejects tokens_only_on_success and ttft_is_total_time.
    clock = Clock()
    plans = {
        "a b c d e": Plan(["1", "2", "3", "4", "5", "6"], [0.5, 0.25, 0.25, 0.25, 0.25, 0.25]),
        "": Plan(["solo"], [0.125]),
        "nothing to say": Plan([]),
        "fails first": Plan(["x"], fail_at=0, fail_delay=4.0),
        "fails  after   three": Plan(["x", "y", "z", "w"], [0.0625, 2.0, 2.0], fail_at=3),
        "fails at the end": Plan(["x", "y"], [3.0, 0.5], fail_at=2),
        "limit before failure": Plan(["x", "y", "z"], [0.75], fail_at=2),
    }
    model = Model(clock, plans)
    expected = Expected()
    with serve(ctx, model, clock) as client:
        generate(client, model, expected, "a b c d e")
        generate(client, model, expected, "a b c d e", 2)
        generate(client, model, expected, "a b c d e", 1)
        generate(client, model, expected, "")
        generate(client, model, expected, "nothing to say")
        check_scrape(client, expected, "after five successful requests")
        generate(client, model, expected, "fails first")
        generate(client, model, expected, "fails  after   three")
        generate(client, model, expected, "fails at the end")
        generate(client, model, expected, "limit before failure", 1)
        check_scrape(client, expected, "after failures at the first, a middle, and the last piece")


@case("invalid bodies are counted as 400 and never reach the model")
def bad_requests(ctx):
    # Rejects bad_requests_not_counted.
    clock = Clock()
    model = Model(clock, {"ok": Plan(["fine"], [0.375])})
    expected = Expected()
    bodies = [
        b"{not json",
        b"",
        b"[]",
        b"{}",
        b'{"prompt": null}',
        b'{"prompt": "ok", "max_tokens": 0}',
        b'{"prompt": "ok", "max_tokens": "3"}',
        b'{"prompt": "ok", "max_tokens": 2.5}',
        b'{"prompt": "ok", "max_tokens": true}',
    ]
    with serve(ctx, model, clock) as client:
        for body in bodies:
            bad_request(client, expected, body)
        check.that(
            model.inner.calls == [],
            "the model was called for an invalid request",
            f"prompts: {[call.prompt for call in model.inner.calls]!r}\n",
        )
        check_scrape(client, expected, "after nine invalid requests")
        generate(client, model, expected, "ok")
        check_scrape(client, expected, "after a valid request following the invalid ones")


@case("a seeded workload of 60 mixed requests, scraped along the way")
def mixed_workload(ctx):
    rng = ctx.rng
    clock = Clock(start=float(rng.integers(0, 2**20)))
    plans = {}
    for i in range(12):
        n = int(rng.integers(0, 7))
        delays = [float(rng.integers(0, 3072)) / 1024 for _ in range(n)]
        fail_at = int(rng.integers(0, n + 1)) if rng.random() < 0.3 else None
        prompt = " ".join(["w"] * int(rng.integers(0, 9))) + f" #{i}"
        plans[prompt] = Plan([f"t{k}" for k in range(n)], delays, fail_at)
    prompts = list(plans)
    model = Model(clock, plans)
    expected = Expected()
    with serve(ctx, model, clock) as client:
        for step in range(60):
            if rng.random() < 0.15:
                bad_request(client, expected, b'{"prompt": 1}')
            else:
                prompt = prompts[int(rng.integers(0, len(prompts)))]
                max_tokens = int(rng.integers(1, 8)) if rng.random() < 0.5 else None
                generate(client, model, expected, prompt, max_tokens)
            if step % 20 == 19:
                check_scrape(client, expected, f"after {step + 1} requests")
