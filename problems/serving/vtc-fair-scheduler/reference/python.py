import asyncio


class VTCScheduler:
    def __init__(self, engine, max_running: int, w_in: float = 1.0, w_out: float = 2.0):
        self.engine = engine
        self.max_running = max_running
        self.w_in = w_in
        self.w_out = w_out
        self.counters = {}  # tenant -> virtual token counter: weighted service received
        self._waiting = []  # the queue Q: requests without a slot, in arrival order
        self._running = []  # ids of live sequences, in admission order
        self._tenant = {}  # id of a live sequence -> its tenant
        self._last = None  # tenant of the last request that left the queue
        self._wake = asyncio.Event()

    def submit(self, request) -> None:
        tenant = request.tenant
        counter = self.counters.setdefault(tenant, 0.0)
        queued = {r.tenant for r in self._waiting}
        if tenant not in queued:
            # The tenant is new to the queue. Whatever it did not use while it was away
            # is gone: bring it up to the least-served tenant that is queueing, or, if
            # nobody is, to the tenant served last. `max`: a lift never lowers.
            if queued:
                counter = max(counter, min(self.counters[t] for t in queued))
            elif self._last is not None:
                counter = max(counter, self.counters[self._last])
            self.counters[tenant] = counter
        self._waiting.append(request)
        self._wake.set()

    def _pick(self):
        # The earliest request of the queued tenant with the smallest counter. `min`
        # keeps the first of equal keys, so ties go to the request first in the queue.
        request = min(self._waiting, key=lambda r: self.counters[r.tenant])
        self._waiting.remove(request)
        return request

    async def run(self) -> None:
        while True:
            if not self._waiting and not self._running:
                self._wake.clear()
                await self._wake.wait()
                continue

            decode = list(self._running)
            prefill = []
            while self._waiting and len(self._running) < self.max_running:
                request = self._pick()
                # Input tokens are charged on admission, at once, so the next pick of
                # this same step already sees them.
                self.counters[request.tenant] += self.w_in * request.prompt_tokens
                self._last = request.tenant
                self._tenant[request.id] = request.tenant
                self.engine.admit(request)
                prefill.append((request.id, request.prompt_tokens))
                self._running.append(request.id)

            result = await self.engine.step(prefill=prefill, decode=decode)

            # Output tokens are charged one by one as they are generated.
            for seq_id in result.tokens:
                self.counters[self._tenant[seq_id]] += self.w_out
            for seq_id in result.finished:
                self._running.remove(seq_id)
                del self._tenant[seq_id]
