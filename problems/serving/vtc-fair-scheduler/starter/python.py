import asyncio


class VTCScheduler:
    """A continuous-batching scheduler that is fair between tenants.

    As given, this class is a complete first-come-first-served scheduler: the loop,
    the slots, and the wake-up are done. The Virtual Token Counter rules are missing.
    """

    def __init__(self, engine, max_running: int, w_in: float = 1.0, w_out: float = 2.0):
        """engine: a `SimEngine` with `max_running` slots.
        w_in, w_out: the weight of one input (prompt) token and of one output token.
        """
        self.engine = engine
        self.max_running = max_running
        self.w_in = w_in
        self.w_out = w_out
        # tenant -> virtual token counter (a float). The tests read this dict. A tenant
        # appears in it, with 0.0, from its first `submit`, and is never removed.
        self.counters = {}
        self._waiting = []  # the queue Q: requests without a slot, in arrival order
        self._running = []  # ids of live sequences, in admission order
        self._wake = asyncio.Event()

    def submit(self, request) -> None:
        """A request arrives (the paper's monitoring stream). Called at the arrival
        instant from a plain callback; it must not block.

        request: a `SimRequest`; `request.tenant` is a string. Apply the counter lift
        if the tenant has no request in the queue, then append the request.
        """
        self._waiting.append(request)
        self._wake.set()

    def _pick(self):
        """Remove and return the request that gets the next free slot.

        VTC: the earliest request of the queued tenant with the smallest counter.
        """
        return self._waiting.pop(0)

    async def run(self) -> None:
        """The scheduling loop (the paper's execution stream); it never returns."""
        while True:
            if not self._waiting and not self._running:
                self._wake.clear()
                await self._wake.wait()  # idle: block until `submit` wakes the loop
                continue

            decode = list(self._running)
            prefill = []
            while self._waiting and len(self._running) < self.max_running:
                request = self._pick()
                self.engine.admit(request)
                prefill.append((request.id, request.prompt_tokens))
                self._running.append(request.id)

            result = await self.engine.step(prefill=prefill, decode=decode)

            # result.tokens: ids that produced one output token in this step.
            for seq_id in result.finished:
                self._running.remove(seq_id)
