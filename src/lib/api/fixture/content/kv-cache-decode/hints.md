Count the calls for the three-token example by hand. Where does a loop that calls `forward` at the top of every iteration spend one token too many?
---
Call `forward` on the prompt once, before the loop. Inside the loop: pick a token from the logits you already have, append it, stop if it is the end-of-sequence token or the budget is spent, and only then call `forward` for the next step.
---
Feed `forward` only the single new token, shaped `(1, 1)`, together with the cache from the previous call. Never concatenate the history yourself; the cache is the history.
