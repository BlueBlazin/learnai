The policy is short; the performance requirement is the real content.

```python
from collections import deque

class Scheduler:
    def __init__(self, max_batch: int):
        self.max_batch = max_batch
        self.waiting: deque[int] = deque()
        self.running: dict[int, None] = {}     # insertion-ordered set

    def submit(self, request_id, tokens_needed):
        self.waiting.append(request_id)

    def next_batch(self):
        while self.waiting and len(self.running) < self.max_batch:
            self.running[self.waiting.popleft()] = None
        return list(self.running)

    def on_step_done(self, finished):
        for request_id in finished:
            del self.running[request_id]
```

A `dict` gives ordered iteration and $O(1)$ deletion, which a list does not. The scheduler never needs `tokens_needed`: the simulator reports completions, as a real engine reports end-of-sequence.
