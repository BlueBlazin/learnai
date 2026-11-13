Work entirely in sorted order and map back at the end.

```python
probs = torch.softmax(logits, dim=-1)
sorted_probs, order = probs.sort(dim=-1, descending=True, stable=True)
before = sorted_probs.cumsum(dim=-1) - sorted_probs
sorted_probs = torch.where(before < p, sorted_probs, 0.0)
cdf = (sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)).cumsum(dim=-1)
rank = (cdf <= u[:, None]).sum(dim=-1).clamp(max=logits.shape[-1] - 1)
return order.gather(-1, rank[:, None]).squeeze(-1)
```

## The off-by-one

`cumsum >= p` marks the crossing token for removal; `cumsum - probs >= p` marks only what comes after it. With probabilities $[0.5, 0.3, 0.15, 0.05]$ and $p = 0.9$ the first rule keeps two tokens with mass $0.8$, which is less than the $0.9$ the definition demands.

## The clamp

Float32 cumulative sums can end at $0.99999994$. A uniform above that would index one past the end; the clamp sends it to the last nucleus token.
