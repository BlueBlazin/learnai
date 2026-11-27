Structure your solution as two nested loops and assign every variable to one of three scopes: once per call, once per canvas, or once per step. The prefix cache, the current canvas, the previous step's scaled logits, and the list of drafts each belong to exactly one scope. Two of the most common bugs come from letting a canvas-level variable leak into the next canvas.

---

Each step produces three `(L,)` token tensors that are easy to mix up: the sampled candidates, the canvas after acceptance and renoising, and the argmax draft. Feed only the renoised canvas into the next `denoise` call. Use only the argmax draft for stability comparisons, output appending, EOS checking, and encoding into the cache.

Apply the `entropy-bound-denoise-step` logic to the temperature-scaled logits `z = logits / t`. Use `z` for the softmax probabilities, `ops.eb_accept(z, entropy_bound)` for acceptance, `argmax` for the draft, and the mean entropy for confidence. Save `z` as the third argument to the next `denoise` call within the same canvas.

---

To check stability, keep a list of the argmax drafts from earlier steps of the current canvas. A step is stable when at least $s$ prior drafts exist and all of the last $s$ drafts equal the new draft. Write the slice carefully so that $s = 0$ selects an empty slice: `drafts[-s:]` returns the entire list when `s == 0`, whereas `drafts[len(drafts) - s:]` returns `[]`.

Check the stopping condition after updating the canvas and computing the current step's draft. That way, the `draft` variable always holds the final step's draft whether the inner loop breaks early or runs to `max_steps`.

---

When a canvas finishes, search only its final draft for `model.eos_id`. If it appears, replace all tokens after its first occurrence with `model.pad_id`, append the result, and return immediately without calling `model.encode`. Otherwise, append the draft and call `model.encode(draft, cache)` only if another canvas will follow. Trace a 3-canvas run by hand to verify you make exactly 3 encoder calls of sizes $P$, $L$, and $L$.
