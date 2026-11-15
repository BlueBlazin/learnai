If the buffers always have `max_len` slots, what tells the attention how many of them
hold tokens? And if `pos` must stay on the device, which ways of saying "slots `pos`
to `pos + T - 1`" are closed to you, and which operation takes the slots to write as
a tensor of indices instead of as Python numbers?

---

Allocate the buffers in the constructor with a list comprehension (one `torch.zeros`
per layer, with `device=` and `dtype=`), and also one tensor of slot numbers,
`torch.arange(max_len, device=device)`. `T` is `k.shape[2]`, a Python int that costs
nothing. The slots to write are `pos + slots[:T]`, a tensor; `index_copy_` along
axis 2 writes into exactly those slots, in place. The mask compares the row of all
slot numbers with a column holding, for each new token, the last slot it may see.
`reset` zeroes each buffer with an in-place method.

---

```python
index = pos + self._slots[: k.shape[2]]
self.k[layer].index_copy_(2, index, k)
self.v[layer].index_copy_(2, index, v)
return self.k[layer], self.v[layer]
```

and for the mask

```python
last_visible = pos + self._slots[:n_new]  # (n_new,)
return self._slots[None, :] <= last_visible[:, None]  # (n_new, max_len)
```

The trailing underscore matters: `index_copy` without it returns a new tensor and
leaves the buffer as it was.
