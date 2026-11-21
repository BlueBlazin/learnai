## The idea

A block text-diffusion model writes $L$ tokens at a time. Within a block it is not
autoregressive: the denoiser sees the whole canvas at once and refines all of it over a
few steps. Across blocks it is exactly as autoregressive as any decoder: the next
canvas is conditioned on everything before it, through a cache that an encoder pass
fills. So the loop is two loops, and each variable lives at one level.

| lives for | what | why |
|---|---|---|
| the call | the text, the prefix cache | the cache only grows, by one finished canvas at a time |
| one canvas | the drafts so far, the previous step's scaled logits | a new canvas starts from noise and owes nothing to the last one |
| one step | temperature, scaled logits, candidates, acceptance, draft | the step of `entropy-bound-denoise-step` |

Inside a step three canvases exist. The *sampled* candidates $y$ are one draw per
position. The *next canvas* $x$ takes the candidates at accepted positions and fresh
noise elsewhere; it is the denoiser's next input and nothing else. The *draft* $d$ is
the argmax at every position, the model's best current guess. The draft is what is
compared from step to step, what is appended, what is searched for the end of the text,
and what the encoder reads. When a canvas converges, all three are the same tokens. When
it runs out of steps they are not, and the choice between them is the difference between
a slightly uncertain block and a block with noise in it, which every later canvas then
reads from the cache.

Two signals stop a canvas, and neither is enough alone. A repeated draft says the guess
did not move, which also happens while the model is still unsure and the temperature
is high. A low mean entropy says the model is sure, which says nothing about whether
it was sure of the same tokens a step ago. With `stability = 0` the first test is
vacuous and a canvas stops at its first confident step.

The cost follows from the levels. The encoder reads the prompt once and each draft
once, and only if another canvas will read it: $P + (g - 1) L$ tokens for $g$ canvases.
The denoiser runs once per step.

## Reference solution (PyTorch)

```python
def diffusion_generate(
    model: Any,
    prompt: list[int],
    cfg: dict[str, Any],
    uniforms: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> list[int]:
    u_init, u_sample, u_noise = uniforms
    length, steps, vocab = cfg["canvas_length"], cfg["max_steps"], model.vocab_size
    out = list(prompt)
    cache = model.encode(prompt)  # the prompt goes through the encoder once

    for c in range(cfg["max_canvases"]):
        canvas = torch.floor(u_init[c] * vocab).long()  # a canvas starts as noise
        previous = None  # self-conditioning starts over with every canvas
        drafts = []  # the argmax canvases of this canvas's steps
        for j in range(steps):
            remaining = steps - j  # counts this step: the first step is the hottest
            temperature = cfg["t_min"] + (cfg["t_max"] - cfg["t_min"]) * remaining / steps
            scaled = model.denoise(canvas, cache, previous) / temperature

            # The step of entropy-bound-denoise-step, on the scaled logits.
            log_p = torch.log_softmax(scaled, dim=-1)
            p = torch.exp(log_p)
            cdf = torch.cumsum(p, dim=-1)
            sampled = (cdf <= u_sample[c, j][:, None] * cdf[:, -1:]).sum(dim=-1)
            accepted = ops.eb_accept(scaled, cfg["entropy_bound"])
            noise = torch.floor(u_noise[c, j] * vocab).long()
            canvas = torch.where(accepted, sampled, noise)  # the next step's input
            draft = torch.argmax(scaled, dim=-1)

            # Stop when the draft repeated the last `stability` drafts and the
            # distributions are sharp. The slice is empty when stability is 0.
            recent = drafts[len(drafts) - cfg["stability"] :]
            stable = len(drafts) >= cfg["stability"] and all(
                torch.equal(draft, old) for old in recent
            )
            confident = float(-(p * log_p).sum(dim=-1).mean()) < cfg["confidence"]
            drafts.append(draft)
            previous = scaled  # what the next step is conditioned on
            if stable and confident:
                break

        # The text gets the draft of the last step, never the sampled canvas.
        final = draft.tolist()
        if model.eos_id in final:
            cut = final.index(model.eos_id) + 1
            return out + final[:cut] + [model.pad_id] * (length - cut)
        out += final
        if c + 1 < cfg["max_canvases"]:  # the last canvas is never read again
            cache = model.encode(draft, cache)
    return out
```

The stop test runs after the sampler and after the draft is recorded, so the `draft`
held when the inner loop ends is the right one for both exits.

## Common mistakes

**What is appended and encoded**

- **The sampled canvas appended.** Invisible on a canvas that converged.
- **The noisy canvas encoded.** The text is right for that canvas and every later
  canvas is conditioned on tokens that are not in the text.
- **No canvas encoded at all**, so each canvas is written as if it followed the prompt.

**Self-conditioning**

- **Left out.** The toy model, like the real one, converges differently without it.
- **The raw logits passed.** The library hands over the logits *after* its logits
  processors, that is, after the division by the temperature.
- **Carried into the next canvas.** It belongs to a canvas; the first step of each
  starts with none.

**Stopping**

- **Stability alone, or confidence alone.**
- **Only the previous draft compared**, which is the rule for `stability = 1` and stops
  one step early for 2.
- **`drafts[-s:]`** for the recent drafts: with `s = 0` that is every draft.
- **Drafts kept across canvases.** The first step of a canvas then counts as a repeat
  if its draft equals the last draft of the canvas before, which a repetitive text
  produces.
- **The sampled canvases compared.** The library's docstring says "accepted canvas";
  its code compares `argmax_canvas`.
- **The entropies summed, or taken from the raw logits.** The threshold is on the mean
  entropy of the scaled distribution, the one the tokens are sampled from.
- **No early stop.** Often the same tokens, at several times the cost.

**The step** (the rules of `entropy-bound-denoise-step`, inside the loop)

- **Steps remaining counted without the current step**, which shifts the whole
  schedule down by one step.
- **Acceptance computed on the raw logits**, and **rejected positions keeping their
  token** instead of being renoised.
- **Uniforms read at the wrong place**: every canvas using canvas 0's rows, or a canvas
  that starts from padding instead of `u_init`.

**The end of the text**

- **No cut after the end-of-sequence token**, a cut that removes the token itself, or
  no check at all.
- **The token searched in the whole text.** A prompt may contain it; only a generated
  one ends the text.

**Cost**

- **The whole text encoded again per canvas.** Right tokens, $O(g^2 L)$ encoder work.
- **The last canvas encoded**, or the canvas that ended the text. Nothing reads it.
- **The prompt encoded token by token**: the right number of tokens in $P$ calls.
- **Building the text in the caller's prompt list.** `out = prompt` and then `out +=`
  leaves the caller with a prompt that has grown.
- **The special tokens written as 1 and 0.** They are the model's, `model.eos_id` and
  `model.pad_id`; one model in the tests ends its text with token 28 and uses token 1
  as ordinary text.
- **Editing the counters, or working on an uncounted copy.** See the next section.

## What the cost judge can and cannot enforce

The counts are the model's own, so the tests hand the model over behind a guard. Code
in the solution file may use `encode`, `denoise`, `vocab_size`, `eos_id`, and `pad_id`.
Any other attribute access on the model (`counters`, `fw`, `seed`, `sharpness`,
`with_framework`, anything private), any attribute set, `copy`, `deepcopy`, and pickling
are refused and recorded, so catching the exception does not help. The model accepts
only caches that it returned itself, so a second `BlockDenoiser` built in the solution
cannot do work on its behalf; its weights are readable only while its own `encode` or
`denoise` runs, so the kit's arithmetic cannot be borrowed as an unbound function; and
the tests keep their own log of the work, which must agree with the counters.
`BlockDenoiser.with_framework` is also forbidden by the harness.

That stops shortcuts, not a determined program. The solution runs in the judge's
process, and Python keeps nothing private there: a solution that imports the guard's
own module, patches the model's class, or walks the objects the garbage collector
tracks can reach the counters, the tests' log, or the weights. The tests do not try to
stop that. The numbers are a check on honest work.

## Complexity

For $g$ canvases with $n_c$ steps each: $g$ encoder calls over $P + (g - 1) L$ tokens
and $\sum_c n_c$ denoiser calls of $L$ tokens, each of which attends to the whole
prefix. A step itself is $O(L V)$ for the softmax and $O(L \log L)$ for the ranking.
With DiffusionGemma's defaults that is at most 48 denoiser passes per 256 tokens, and
as few as 2.

## In the wild

Checked on 2026-11-21 by reading and running `transformers` 5.23.0
(`models/diffusion_gemma/generation_diffusion_gemma.py`).

- **The loop.** `DiffusionGemmaGenerationMixin.generate` encodes "all previous tokens"
  at the top of each canvas iteration (the whole prompt the first time, the last
  `canvas_length` tokens afterwards), draws a canvas with
  `EntropyBoundSampler.initialize_canvas`, resets the stopping criteria, and runs
  `_denoising_step` for `cur_step` from `max_denoising_steps` down to 1. It then
  concatenates `argmax_canvas`, lets `_finalize_canvas` look for the end-of-sequence
  token in the last `canvas_length` tokens and pad after it, and breaks or goes around
  again. Because the encoder call sits at the top of the loop, the last canvas is never
  encoded.
- **The step.** `LinearTemperatureScheduleLogitsProcessor` divides by
  `t_min + (t_max - t_min) * cur_step / max_denoising_steps`, with `cur_step` the steps
  remaining; the canvas is sampled from the softmax of the processed logits and
  `new_argmax_canvas` is their argmax; the sampler accepts and renoises; the processed
  logits become `self_conditioning_logits`.
- **The stop.** `StableAndConfidentStoppingCriteria` keeps the last
  `stability_threshold` argmax canvases, initialized to -1 and reset per canvas, and
  returns `stable & confident` with `confident = mean(entropy) < confidence_threshold`.
  A threshold of 0 makes every step stable.
- **The defaults** (`_get_default_generation_params`): 48 steps, `t_min` 0.4, `t_max`
  0.8, entropy bound 0.1, stability 1, confidence 0.005, 256 new tokens. The tests use
  those thresholds with 12 steps, and a second, looser set (bound 1.0, stability 2,
  confidence 0.5, temperatures 1.0 to 0.2) under which canvases stop while positions
  are still unsettled. The canvas length of 256 for `google/diffusiongemma-26B-A4B-it`
  is from this project's curriculum notes, not read again here.
- **What was run.** `fixtures/make.py` calls the library's `generate` on a small class
  whose encoder and decoder are the kit's `BlockDenoiser`, with `torch.multinomial` and
  `initialize_canvas` replaced by the supplied uniforms, for all 26 scenarios of the
  tests. Tokens, the sizes of the encoder calls, and the number of decoder calls agree
  with the loop of this problem in every one.

**What that does not verify.** `BlockDenoiser` is a toy with the interface of the
shipped model and nothing of its arithmetic: it was not compared with the DiffusionGemma
denoiser, and how often real canvases converge, how many steps they take, and how much
self-conditioning matters are properties of the toy here. The comparison covers the
loop around the model.

Deliberate simplifications, all of them: one sequence (the library batches, freezes
the rows whose canvas has stopped, and pads whole canvases of finished rows);
`max_canvases` instead of `max_new_tokens` (the library runs
`ceil(max_new_tokens / canvas_length)` canvases) and no `max_length` criterion; a single
end-of-sequence id (the library accepts a list); padding always applied (the library
pads only when `pad_token_id` is set); the stopping rule always on (the library turns it
off when either threshold is unset); uniforms as inputs in place of `torch.multinomial`
and `torch.randint`; no streamer, no caller-supplied starting canvas or
self-conditioning logits, no images, no compiled or static cache; canvases of at most 8
tokens and at most 12 steps.
