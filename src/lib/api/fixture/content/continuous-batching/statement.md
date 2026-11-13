A static batcher waits for a batch to fill, runs it to completion, and only then admits new requests. Short requests wait behind long ones, and finished slots sit idle. **Continuous batching** admits a waiting request into the running batch the moment a slot frees up.

Implement the scheduler. Time is virtual: the judge runs a discrete-event simulation and never sleeps.

## Model

The engine executes decode **steps**. One step advances every running request by one token and takes

$$
t_{\text{step}}(b) = t_0 + b \cdot t_1
$$

virtual milliseconds for a batch of $b$ requests, with $t_0 = 20$ and $t_1 = 2$. A request $r$ arrives at time $a_r$ and needs $n_r$ tokens.

## Interface

```python
class Scheduler:
    def __init__(self, max_batch: int): ...

    def submit(self, request_id: int, tokens_needed: int) -> None:
        """Called by the simulator when a request arrives."""

    def next_batch(self) -> list[int]:
        """Called before each step. Return the ids to run this step (at most max_batch)."""

    def on_step_done(self, finished: list[int]) -> None:
        """Called after each step with the ids that produced their last token."""
```

## Rules

1. A running request stays in the batch until it finishes; preemption is not allowed.
2. Free slots are filled first come, first served.
3. `next_batch` returns an empty list only when nothing is waiting or running.

### Example 1

`max_batch = 2`. Three requests arrive at $t = 0$ needing 1, 3, and 2 tokens.

| Step | Batch | Step time | Finishes | Clock after |
|---:|---|---:|---|---:|
| 1 | A, B | 24 ms | A | 24 |
| 2 | B, C | 24 ms | | 48 |
| 3 | B, C | 24 ms | B, C | 72 |

C starts at step 2, as soon as A's slot frees. A static batcher would start C at step 4, after B finishes, and end at $t = 112$.

## How it is judged

The simulator replays seeded arrival traces and compares against the reference policy.

| Metric | Limit |
|---|---|
| Mean time to first token | within 1% of reference |
| Makespan | within 1% of reference |
| Rule violations | 0 |
| Scheduler overhead | `next_batch` is $O(\text{max\_batch})$ amortized; traces have up to $10^5$ requests |

## Constraints

- $1 \le$ `max_batch` $\le 64$; $1 \le n_r \le 2048$.
- Pure Python. No threads, no `time.sleep`, no `asyncio`.
