```python
values, indices = (logits / temperature).topk(k, dim=-1)
return torch.zeros_like(logits).scatter(-1, indices, torch.softmax(values, dim=-1))
```

Taking the softmax over only the $k$ survivors is the same as masking the rest to $-\infty$, and it avoids a pass over the full vocabulary.
