Two masks, combined by broadcasting:

```python
tri = torch.ones(max_len, max_len, dtype=torch.bool).tril()        # (T, T)
keys = torch.arange(max_len) < lengths[:, None]                    # (B, T)
return (tri[None] & keys[:, None, :])[:, None]                     # (B, 1, T, T)
```

The liveness property holds because every length is at least $1$: key $0$ is always valid and always in the past.
