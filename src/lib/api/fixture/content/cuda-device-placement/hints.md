Every factory function takes `device=`. A tensor created without it starts on the CPU.

---

`torch.full` builds the matrix of $-\infty$ where you ask for it; `torch.triu` keeps the strict upper triangle and stays on that device.
