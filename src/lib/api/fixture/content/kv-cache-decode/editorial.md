The structure that hits the optimal count puts the stop check between sampling and the next forward pass.

```python
logits, cache = model.forward(prompt, None)
out = prompt
for step in range(max_new_tokens):
    token = logits[:, -1].argmax(dim=-1, keepdim=True)
    out = torch.cat((out, token), dim=1)
    if token.item() == eos_id or step == max_new_tokens - 1:
        break
    logits, cache = model.forward(token, cache)
return out
```

## Accounting

The prompt costs $L$. Each generated token except the last is fed back once, costing $n - 1$. Total $L + n - 1$.

A loop that calls `forward` first and samples second processes the final token for logits nobody reads. One wasted token per request is small; at a thousand requests a second it is a thousand forward passes a second.
