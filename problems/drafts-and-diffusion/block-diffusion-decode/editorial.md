## The idea

A block text-diffusion model generates $L$ tokens at a time. Within a single block, it is non-autoregressive. The denoiser views the entire canvas at once and refines all $L$ positions together over multiple steps. Across blocks, however, generation is autoregressive. Each new canvas conditions on all earlier tokens through a prefix cache filled by the encoder.

This structure maps directly to two nested loops, where each piece of state belongs to one scope:

| scope | state | why |
|---|---|---|
| entire call | output token list, prefix cache | the cache only grows, by one finished canvas at a time |
| one canvas | list of drafts, previous step's scaled logits | each new canvas starts from fresh noise and does not reuse state from the previous canvas |
| one step | temperature, scaled logits, candidates, acceptance mask, draft | the per-step computation of `entropy-bound-denoise-step` |

Inside each step, three `(L,)` token tensors exist:
- The *sampled candidates* $y$ are one inverse-CDF draw per position.
- The *next canvas* $x$ keeps $y$ at accepted positions and places fresh noise elsewhere. It serves only as the next `denoise` call's input.
- The *draft* $d$ is the per-position argmax of the scaled logits, representing the model's best current prediction.

The draft $d$ is the tensor you compare across steps, append to the output, search for `eos_id`, and encode into the cache. When a canvas converges, $y$, $x$, and $d$ are identical. When it hits `max_steps`, they differ, and encoding $x$ instead of $d$ injects unaccepted noise tokens into the prefix cache for every later canvas.

Early stopping requires both stability and confidence because neither condition suffices alone. A repeated draft only shows that the argmax did not change, which can happen while the distribution is still flat at high temperature. Low mean entropy shows that the model is confident on the current step, but not that its prediction has settled across steps. When `stability = 0`, the stability check is trivially true, so the canvas stops on its first confident step.

The cost bounds follow from these loop scopes. The encoder processes the prompt once ($P$ tokens) and each finished draft once if another canvas follows ($L$ tokens each), totaling $g$ calls and $P + (g - 1) L$ tokens for $g$ canvases. The denoiser runs once per active step.

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

Checking the stopping condition after updating `draft` ensures that `draft` holds the last step's argmax whether the inner loop breaks early or exhausts `max_steps`.

## Common mistakes

**Appending and encoding**

- **Appending the sampled canvas.** This bug is hidden whenever a canvas converges, and only surfaces when a canvas hits `max_steps`.
- **Encoding the noisy canvas instead of the draft.** The current canvas's output tokens are right, but every subsequent canvas conditions on noise tokens that never appeared in the output.
- **Never encoding finished canvases.** Every canvas is then generated as if it immediately followed `prompt`.

**Self-conditioning**

- **Omitting `previous`.** Both the toy model and DiffusionGemma converge to different trajectories without self-conditioning.
- **Passing raw unscaled logits as `previous`.** The model conditions on the logits *after* dividing by temperature.
- **Carrying `previous` across canvases.** Self-conditioning resets to `None` on step 0 of every canvas.

**Stopping criteria**

- **Checking only stability or only confidence.** Both conditions must hold together.
- **Comparing only the single previous draft.** That works for `stability = 1`, but stops one step too early when `stability = 2`.
- **Slicing with `drafts[-s:]`.** When stability `s = 0`, `drafts[-0:]` returns the entire list instead of an empty slice. Use `drafts[len(drafts) - s:]`.
- **Keeping `drafts` across canvases.** On repetitive text, step 0 of a new canvas can match the final draft of the previous canvas and trigger a false stability match.
- **Comparing sampled canvases instead of argmax drafts.** Even though the Hugging Face docstring mentions "accepted canvas", the implementation compares `argmax_canvas`.
- **Summing entropies instead of averaging, or computing entropy on raw logits.** Confidence compares the *mean* entropy of the temperature-scaled distribution against `confidence`.
- **Omitting early stopping.** This often produces the same tokens at several times the denoiser call count.

**Per-step mechanics**

- **Excluding the current step from `remaining`.** Using `steps - j - 1` instead of `steps - j` shifts the entire temperature schedule down by one step.
- **Computing acceptance on raw logits, or keeping old tokens at rejected positions.** Acceptance uses the scaled logits, and rejected positions must be renoised from `u_noise`.
- **Indexing uniforms incorrectly.** Reusing canvas 0's uniform rows across canvases, or initializing a canvas from `pad_id` instead of `u_init`, breaks exact token matching.

**End-of-sequence handling**

- **Failing to pad after `eos_id`, dropping `eos_id` itself, or skipping the EOS check.** Keep the first `eos_id` and replace all following tokens in that canvas with `pad_id`.
- **Searching the entire output instead of the current draft.** The prompt may contain `eos_id`, so only the newly generated draft can terminate generation.

**Cost and interface**

- **Re-encoding the entire output on every canvas.** This produces the right tokens but does $O(g^2 L)$ encoder work instead of $P + (g - 1) L$.
- **Encoding the last canvas or the EOS canvas.** No further canvas reads it.
- **Encoding the prompt one token at a time.** This encodes $P$ tokens in $P$ calls instead of 1 call.
- **Mutating the caller's `prompt` list.** Writing `out = prompt` followed by `out += ...` modifies the caller's list in place.
- **Hardcoding `1` and `0` for EOS and padding.** Always read `model.eos_id` and `model.pad_id`. One test uses token `28` for EOS and token `1` as an ordinary token.
- **Modifying counters or cloning the model.** See the next section.

## What the cost judge can and cannot enforce

The tests wrap `model` in a guard so that solution code can access only `encode`, `denoise`, `vocab_size`, `eos_id`, and `pad_id`. Any other attribute access (such as `counters`, `fw`, `seed`, `sharpness`, `with_framework`, or private attributes), attribute assignment, copying, or pickling is refused and logged, so catching the exception does not bypass the check. The model accepts only caches it created itself, exposes its weights only inside its own `encode` and `denoise` calls, and cross-checks its counters against a separate test log. `BlockDenoiser.with_framework` is also forbidden by the harness.

Because the solution runs in the same Python process as the judge, these guards prevent accidental shortcuts rather than adversarial tampering (such as importing the guard module, monkey-patching classes, or inspecting objects via `gc`). The counter checks are designed to verify honest implementations.

## Complexity

Generating $g$ canvases with $n_c$ denoising steps for canvas $c$ requires $g$ encoder calls covering $P + (g - 1) L$ total tokens and $\sum_{c=0}^{g-1} n_c$ denoiser calls of $L$ tokens each (with each call attending to the current prefix cache). Outside the model, each step takes $O(L V)$ time for softmax and inverse-CDF sampling (with vocabulary size $V$) and $O(L \log L)$ time for sorting entropies. With DiffusionGemma's default settings, a 256-token canvas takes between 2 and 48 denoiser passes.

## In the wild

Verified on 2026-11-21 against `transformers` 5.23.0 (`models/diffusion_gemma/generation_diffusion_gemma.py`):

- **Outer loop.** `DiffusionGemmaGenerationMixin.generate` encodes the newly finalized prefix at the top of each canvas iteration (the full prompt on the first iteration, and the previous `canvas_length` tokens afterward). It initializes each canvas with `EntropyBoundSampler.initialize_canvas`, resets stopping criteria, runs `_denoising_step` for `cur_step` from `max_denoising_steps` down to `1`, appends `argmax_canvas`, and calls `_finalize_canvas` to check for EOS in the last `canvas_length` tokens and pad after it. Placing the encoder call at the top of the loop ensures the final canvas is never encoded.
- **Denoising step.** `LinearTemperatureScheduleLogitsProcessor` divides logits by `t_min + (t_max - t_min) * cur_step / max_denoising_steps`. The sampler draws candidates from the softmax of the processed logits, computes `new_argmax_canvas`, accepts and renoises positions, and stores the processed logits as `self_conditioning_logits`.
- **Stopping rule.** `StableAndConfidentStoppingCriteria` tracks the last `stability_threshold` argmax canvases (initialized to `-1` and reset per canvas) and returns `stable & confident`, where `confident = mean(entropy) < confidence_threshold`. A threshold of `0` marks every step as stable.
- **Defaults.** `_get_default_generation_params` sets 48 steps, `t_min = 0.4`, `t_max = 0.8`, `entropy_bound = 0.1`, `stability = 1`, `confidence = 0.005`, and 256 new tokens. The test suite uses these thresholds with 12 steps, plus a looser preset (`entropy_bound = 1.0`, `stability = 2`, `confidence = 0.5`, temperatures `1.0` down to `0.2`) where canvases stop while positions remain unsettled. The 256-token canvas length for `google/diffusiongemma-26B-A4B-it` comes from this project's curriculum notes.
- **Fixture verification.** `fixtures/make.py` runs `DiffusionGemmaGenerationMixin.generate` on a thin wrapper around `BlockDenoiser`, substituting the supplied uniforms for `torch.multinomial` and `initialize_canvas`, across all 26 test scenarios. Output tokens, encoder call sizes, and denoiser call counts match this specification in every case.

**Scope of verification.** `BlockDenoiser` is a toy model matching the real model's interface rather than its neural arithmetic. Convergence rates, step counts, and the strength of self-conditioning in the tests are properties of the toy model. The cross-check verifies the generation loop around the model.

**Deliberate simplifications** compared to `transformers`:
- Single sequence instead of batched generation (where finished rows are frozen and padded across canvases).
- `max_canvases` instead of `max_new_tokens` (which runs `ceil(max_new_tokens / canvas_length)` canvases) and no `max_length` criterion.
- A single `eos_id` instead of a list of EOS token IDs.
- Padding always applied after EOS (the library pads only when `pad_token_id` is configured).
- Early stopping always enabled (the library disables it when either threshold is unset).
- Explicit uniform tensors in place of `torch.multinomial` and `torch.randint`.
- No streamer, caller-supplied starting canvas, caller-supplied self-conditioning logits, image inputs, or compiled/static cache.
- Small problem sizes (at most 8 tokens per canvas and 12 steps).
