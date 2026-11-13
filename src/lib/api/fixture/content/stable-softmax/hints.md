What happens to $e^{z_i}$ when $z_i = 1000$ in float32? Try it in a REPL before you change anything.
---
Softmax does not change when you subtract the same number from every logit. Which number makes the largest exponent harmless?
---
Subtract the maximum along `dim` (keep the dimension so it broadcasts), exponentiate, and divide by the sum along `dim`. A logit of $-\infty$ becomes $e^{-\infty} = 0$ with no special handling.
