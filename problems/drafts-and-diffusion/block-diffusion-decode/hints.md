Write the loop as two nested loops and decide, for every variable, at which level it
lives: once per call, once per canvas, or once per step. The cache, the current
canvas, the previous step's scaled logits, and the list of drafts each belong to
exactly one level, and two of the bugs this problem is about are a variable kept alive
one level too long.

---

One step produces three things of shape `(L,)` that are easy to confuse: the sampled
candidates, the canvas after acceptance and renoising, and the argmax draft. Only the
second is fed to the next `denoise` call. Only the third is compared between steps,
appended to the text, searched for the end-of-sequence token, and encoded.

The step is the function of `entropy-bound-denoise-step` applied to `z = logits / t`:
its softmax for the sampling, `ops.eb_accept(z, bound)` for the acceptance, its
argmax for the draft, its entropies for the mean. Keep `z` after the step; it is the
third argument of the next `denoise` call of the same canvas.

---

The stop test needs the drafts of the earlier steps of this canvas. Stability asks
whether the last $s$ of them all equal the new draft, and whether there are $s$ of them
at all. Write the slice so that it is empty for $s = 0$: `drafts[-s:]` is the whole list
when `s` is 0, `drafts[len(drafts) - s:]` is not.

Test for stopping after the sampler has run and the draft is recorded; then the draft
you hold when the loop ends is the right one in both cases, a stop and the step limit.

---

After a canvas: look for the end-of-sequence token in its draft only, and if it is
there cut, pad, and return without touching the encoder. Otherwise append the draft, and
call `model.encode(draft, cache)` only if another canvas follows. Count the encoder
calls of your loop by hand for three canvases: there should be three, of $P$, $L$, and
$L$ tokens.
