## The idea

Two classic results from real-time scheduling carry the problem.

**Earliest deadline first is the right order when the work fits.** If any order
meets all deadlines, serving by earliest deadline does. The intuition is an exchange
argument: if a schedule serves a later deadline before an earlier one, swapping the
two never makes the earlier-deadline job later than the other one was, so nothing
that was on time becomes late.

**Under overload, EDF alone is the wrong order.** When not everything fits, the
earliest deadline belongs to the request that is closest to being lost. EDF spends
capacity on it, makes the next one late in turn, and so on: the domino effect. The
remedy is admission control: work that cannot finish in time is rejected before it
consumes anything. Then EDF again runs on a set that fits.

In a batched engine both rules need the step's cost model, because requests admitted
together share one forward pass and one completion time:

$$
f = \text{now} + \text{step\_ms}(P_\text{admitted} + P_r,\ D).
$$

This one number answers both questions.

- *Is $r$ hopeless?* If $f$ is after $r$'s deadline, then yes, for good. $r$ cannot
  be on time in this step. If it waits, the next step starts when this one ends,
  which is no earlier than $\text{now} + \text{step\_ms}(P_\text{admitted}, D)$, and
  it then takes at least a step that prefills $P_r$; the sum is larger than $f$. So
  the shed rule is *sound*: it never refuses a request that some continuation could
  still have served on time. That is why it can be applied at once, also to requests
  for which no slot is free.
- *Does $r$ hurt somebody?* Its prompt lengthens the step for everyone in it. The
  requests already admitted have earlier deadlines (the scan is in deadline order),
  and the first of them has the earliest, so $f \le \text{deadline}(\textit{admit}[0])$
  is the whole check. A request that fails only this check is not hopeless; it
  waits, and the scan goes on, since a smaller prompt behind it may still fit.

There is no preemption. A running sequence has produced its first token, which is
all the TTFT deadline asks of it; evicting it would throw away its prefill for no
gain in the metric this scheduler optimizes.

## Reference solution (Python)

```python
def deadline_of(request):
    return math.inf if request.deadline_s is None else request.deadline_s


def plan_step(waiting, now_ms, running, free_slots, step_ms):
    admit, shed = [], []
    prefill_tokens = 0
    for request in sorted(waiting, key=deadline_of):  # stable: ties keep arrival order
        first_token = now_ms + step_ms(prefill_tokens + request.prompt_tokens, running)
        if first_token > deadline_of(request):
            shed.append(request)
        elif len(admit) < free_slots and (not admit or first_token <= deadline_of(admit[0])):
            admit.append(request)
            prefill_tokens += request.prompt_tokens
    for request in (*admit, *shed):
        waiting.remove(request)
    return admit, shed
```

The loop calls it with `loop.time()`, `len(running)`, `max_running - len(running)`,
and `engine.cost.step_s`, reports the shed requests, and skips the step when nothing
was admitted and nothing is running (everything waiting was shed).

## Common mistakes

- **First come, first served.** With the same shedding rule it is safe, and worse
  where it matters: in a burst, the loose deadlines that happened to arrive first
  take the slots, and the tight ones behind them are shed. On the burst fixture
  attainment falls from 0.84 to 0.72.
- **Serving hopeless requests.** Without shedding, the queue keeps every request
  and serves all of them late. Each late request also delays the ones behind it, so
  under sustained overload attainment collapses to a few percent while the engine
  is fully busy.
- **Shedding only what has already expired.** `now > deadline` is not the question;
  "will the first token be on time" is. A request admitted with 5 ms left and a
  40 ms prefill is served late and takes a slot for its whole generation.
- **An estimate that ignores the requests ahead.** Computing $f$ from the request's
  own prompt alone is right for the first admission and wrong for every further
  one: the step is longer than estimated, and requests judged on time are late.
- **An estimate that ignores the decodes.** The step also decodes every running
  sequence; with a full batch that term is as large as a small prefill.
- **Not protecting the admitted requests.** Checking only $r$'s own deadline lets a
  long prompt with a loose deadline push a short urgent prompt past its deadline in
  the very step that was supposed to save it.
- **`>=` at the boundary.** A first token at the deadline is on time; shedding it
  refuses a request that would have counted.
- **Ties broken by id,** which a heap of `(deadline, id)` tuples does silently. Ids
  are opaque; equal deadlines go by arrival.
- **Stopping the scan** at the first request that has to wait, or once the slots are
  full: requests further back that fit are left out, and hopeless ones are told a
  step later than they could have been.

## Complexity

One sort per step, $O(W \log W)$ for $W$ waiting requests, plus $O(W)$ removals from
a list ($O(W^2)$ in the worst case as written; a deadline-ordered heap with lazy
deletion brings the plan to $O((A + S) \log W)$ for $A$ admissions and $S$ sheds
when the scan can stop early, which this specification's "scan everything" rule
trades away for shedding as early as possible).

## In the wild

- The optimality of deadline-driven scheduling for one processor is the classic
  result of "Scheduling Algorithms for Multiprogramming in a Hard-Real-Time
  Environment" (Journal of the ACM, 1973); the collapse of EDF under overload and
  admission tests as the cure are standard material of real-time systems textbooks.
- **Triton Inference Server** lets a model's queue policy carry a timeout
  (`default_timeout_microseconds`, optionally overridden per request) with
  `timeout_action` `REJECT` or `DELAY`: requests that have waited too long are
  rejected or deferred instead of being batched. That is shedding on expiry, the
  weaker rule from the list above; shedding on *predicted* lateness needs a cost
  model like `step_ms`.
- **vLLM** orders its waiting queue first come, first served or, with
  `--scheduling-policy priority`, by a per-request priority (`priority-scheduler`).
  A deadline can be mapped onto that priority by the gateway in front; the shedding
  then also has to live in the gateway, as `admission-control` does with queue
  depth.
- **DistServe** (OSDI 2024) and the goodput literature evaluate schedulers by
  exactly the metric of the attainment cases: the share of requests that meet both
  TTFT and TPOT objectives, as a function of rate (`goodput-slo`). This problem
  schedules on the TTFT half. The completion half cannot be planned the same way,
  since a server does not know an output's length in advance; it is enforced while
  a request runs, by `deadline-cancellation`.
