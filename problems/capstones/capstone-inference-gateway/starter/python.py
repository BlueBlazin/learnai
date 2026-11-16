import asyncio
from dataclasses import dataclass


@dataclass(frozen=True)
class Rejected:
    """What the future returned by `submit` holds for a request that was not accepted."""

    reason: str  # "rate_limited" or "queue_full"
    retry_after_s: float | None = None
    status: str = "rejected"


class Gateway:
    """An inference gateway in front of the kit's `SimCluster`.

    `cfg` has the attributes listed in the statement. Read the time with
    `cluster.sim.now()` and create futures with `cluster.sim.loop.create_future()`.

    Stage 1: `submit` (admission, the gateway queue, dispatch to a replica).
    Stage 2: which replica (least outstanding tokens).
    Stage 3: which replica, with `cfg.prefix_routing`.
    Stage 4: which request, with `cfg.fair`; `counters` maps tenant -> counter.
    Stage 5: `evaluate`.
    """

    def __init__(self, cluster, cfg):
        self.cluster = cluster
        self.cfg = cfg
        self.counters = {}

    def submit(self, request) -> asyncio.Future:
        """A request has arrived. A plain method, called inside the event loop.

        Returns a future. For a rejected request it is already resolved, with a
        `Rejected`. For an admitted request it resolves with the `Outcome` of the
        replica that served it.
        """
        raise NotImplementedError

    def evaluate(self) -> int:
        """Stage 5: one evaluation of the autoscaling policy.

        Reads `cluster.metrics()`, applies the policy with `cluster.scale_to(n)`,
        dispatches, and returns `n`.
        """
        raise NotImplementedError
