Keep two collections: the requests running now, and a queue of those waiting. Which data structure gives $O(1)$ removal from the front of the queue?
---
`collections.deque` for the waiting queue. `list.pop(0)` is $O(n)$ and is exactly what times out on the large traces.
---
In `on_step_done`, remove the finished ids from the running set. In `next_batch`, top the running set up from the queue until it has `max_batch` entries or the queue is empty, then return it in admission order.
