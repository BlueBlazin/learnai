The direct version recounts every pair on each iteration. It is $O(\text{merges} \times \text{corpus})$ and passes at these sizes.

```python
def merge_word(symbols, pair):
    out, i = [], 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)
```

Production trainers keep the pair counts incrementally and update only the words that contain the merged pair, with a priority queue for the maximum.
