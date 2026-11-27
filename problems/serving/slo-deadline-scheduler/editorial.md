## The idea

Two classic results from real-time scheduling motivate this design.

**Earliest deadline first (EDF) is optimal when all work fits.** If any admission order can meet every deadline, serving requests in earliest-deadline order does so too. Swapping an earlier-deadline request ahead of a later-deadline one never pushes the earlier request past the later deadline, so no on-time request becomes late.

**Under overload, EDF alone collapses.** When load exceeds capacity, the earliest deadline belongs to the request closest to missing its target. Spending capacity on a doomed request delays the next request until it misses too, creating a domino effect. Admission control solves this by rejecting work that cannot finish in time before it consumes any slots, leaving EDF to schedule a feasible subset.

In a batched engine, both rules depend on the step cost model because all requests admitted in a step share one forward pass and one completion time. Let $P_{\text{admitted}}$ be the prompt tokens of requests already admitted to the current step, $P_r$ be the prompt tokens of candidate request $r$, and $D$ be the number of currently running sequences. If $r$ joins the step, every admitted request gets its first token at

$$
f = \text{now} + \text{step\_ms}(P_{\text{admitted}} + P_r,\ D).
$$

This single estimate answers both admission questions:

- **Can $r$ still make its deadline?** If $f$ exceeds $r$'s deadline, $r$ is permanently hopeless. It cannot finish on time in this step. If it waits, the next step cannot start before $\text{now} + \text{step\_ms}(P_{\text{admitted}}, D)$ and must still prefill $P_r$, which finishes even later than $f$. The shed rule is therefore sound and never rejects a request that could have been saved, even when no slot is free right now.
- **Does $r$ break an already-admitted request?** Adding $r$ lengthens the step for everyone in `admit`. Because we scan in earliest-deadline order, `admit[0]` has the tightest deadline in `admit`. Checking $f \le \text{deadline}(\textit{admit}[0])$ is therefore sufficient to protect all admitted requests. If $r$ fails only this check, it is not hopeless and simply waits while the scan continues for smaller prompts behind it.

There is no preemption because a running sequence has already produced its first token. Evicting it would discard its prefill without improving TTFT attainment.

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

The `run` loop calls `plan_step` with `loop.time()`, `len(running)`, `max_running - len(running)`, and `engine.cost.step_s`, notifies `on_shed` for each shed request, and skips the step when nothing is admitted and nothing is running.

## Common mistakes

- **Using first-come-first-served order.** Adding shedding to FCFS prevents late first tokens, but loose-deadline requests that arrive early in a burst consume the slots and force tighter deadlines behind them to be shed. On the burst workload, attainment drops from 0.84 to 0.72.
- **Omitting shedding.** Keeping hopeless requests causes the engine to serve requests after their deadlines, delaying every request behind them. Under sustained overload, attainment collapses to a few percent while the engine stays 100% busy.
- **Shedding only already-expired requests.** Checking `now > deadline` ignores prefill time. A request with 5 ms remaining and a 40 ms prefill will still miss its deadline and occupy a slot for its full generation.
- **Ignoring previously admitted prompts in $f$.** Estimating $f$ from `request.prompt_tokens` alone works only for the first admission in a step. Subsequent admissions underestimate the step duration and produce late first tokens.
- **Ignoring running decodes in $f$.** The step also decodes all `running` sequences. With a full batch, the decode term can be as large as a small prefill.
- **Not protecting already-admitted requests.** Checking only $r$'s own deadline allows a long prompt with a loose deadline to push a short, urgent prompt past its deadline in the same step.
- **Using `>=` instead of `>` for shedding.** A first token arriving at the exact deadline is on time and must not be shed.
- **Breaking ties by request ID.** A heap of `(deadline, id)` tuples silently breaks equal-deadline ties by ID instead of arrival order. Request IDs are opaque and must not affect tie-breaking.
- **Stopping the scan early.** Breaking out of the loop at the first waiting request or when `free_slots` is reached skips smaller prompts behind it that still fit and delays shedding hopeless requests by at least one step.

## Complexity

For $W$ waiting requests, sorting takes $O(W \log W)$ time per step. Removing admitted and shed requests from a Python list takes $O(W)$ per removal, or $O(W^2)$ worst-case time per step (which can be reduced to $O(W)$ by rebuilding `waiting` via a set of decided requests). A deadline-ordered heap with lazy deletion can plan in $O((A + S) \log W)$ time for $A$ admissions and $S$ sheds when early stopping is allowed, though scanning the full queue is required here to shed hopeless requests as early as possible.

## In the wild

- **Real-time scheduling theory.** The optimality of EDF on a single processor was proved in Liu and Layland, "Scheduling Algorithms for Multiprogramming in a Hard-Real-Time Environment" (Journal of the ACM, 1973). The overload collapse of EDF and admission tests as the fix are standard topics in real-time systems.
- **Triton Inference Server** supports per-model queue timeouts (`default_timeout_microseconds`, with per-request overrides) and a `timeout_action` of `REJECT` or `DELAY`. This implements expiry-based shedding, whereas shedding on predicted lateness requires a step cost model like `step_ms`.
- **vLLM** orders its waiting queue FCFS by default or by per-request priority with `--scheduling-policy priority` (see `priority-scheduler`). A gateway can map deadlines to priorities, though load shedding must then also happen in the gateway (as in `admission-control`).
- **DistServe** (OSDI 2024) and related goodput work evaluate schedulers by the fraction of requests meeting both TTFT and time-per-output-token (TPOT) objectives as a function of request rate (see `goodput-slo`). This problem schedules for the TTFT objective. Completion deadlines cannot be checked before admission because output lengths are unknown in advance, so they are enforced during generation (see `deadline-cancellation`).
