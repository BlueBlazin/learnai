Implement the single-sequence generation loop of a block text-diffusion model, matching DiffusionGemma's `generate`. Your function encodes a prompt once into a prefix cache, then extends the text one block (called a *canvas*) of $L$ tokens at a time by denoising from random tokens over several steps and encoding each finished canvas into the cache. The judge checks exact output tokens as well as exact encoder and denoiser call counts.

## Specification

```python
def diffusion_generate(model, prompt, cfg, uniforms): ...  # -> list of ints
```

`model` is a [`BlockDenoiser`](../../../python/src/learnai/kit/diffusion.py) with vocabulary size $V = \texttt{model.vocab\_size}$ and special token IDs `model.eos_id` (end-of-sequence) and `model.pad_id` (padding). It exposes four call forms:

```python
cache = model.encode(tokens)                      # encode a new prefix (list of ints or 1-D tensor)
cache = model.encode(tokens, cache)               # extend an existing prefix cache with tokens
logits = model.denoise(canvas, cache)             # float32 (L, V) logits for the L tokens after cache
logits = model.denoise(canvas, cache, previous)   # same, self-conditioned on float32 (L, V) logits
```

Neither `encode` nor `denoise` modifies the cache passed to it.

The remaining inputs are:
- `prompt`: a non-empty list of $P \ge 1$ Python `int` token IDs.
- `cfg`: a dictionary with keys `canvas_length` ($L$), `max_canvases`, `max_steps` ($N$), `t_min` ($t_{\min}$), `t_max` ($t_{\max}$), `entropy_bound`, `stability` ($s$), and `confidence`.
- `uniforms`: a tuple `(u_init, u_sample, u_noise)` of three float32 tensors with entries in $[0, 1)$. All randomness comes from these tensors.

### Outer loop (generating canvases)

First, encode the prompt once with `cache = model.encode(prompt)` and initialize the output sequence with a copy of `prompt`. Then, for canvas index $c = 0, 1, \dots, \texttt{max\_canvases} - 1$:

1. **Initialize from noise.** Create the starting canvas $x$ of length $L$ with
   $$x_i = \lfloor \texttt{u\_init}[c, i] \cdot V \rfloor, \quad i \in \{0, \dots, L - 1\},$$
   computing the product in float32.
2. **Denoise.** Run the inner denoising loop below for up to $N$ steps. This produces a final *draft* $d$ of $L$ tokens.
3. **Handle EOS and append.** Search only the draft $d$ (never `prompt`) for `model.eos_id`. If `model.eos_id` appears in $d$, replace every token strictly after its first occurrence with `model.pad_id`, append these $L$ tokens to the output, and return immediately without encoding this canvas.
4. **Update cache or finish.** Otherwise, append $d$ to the output. If $c = \texttt{max\_canvases} - 1$, return. Otherwise, update `cache = model.encode(d, cache)` and continue to canvas $c + 1$.

### Inner loop (denoising steps for canvas $c$)

For step $j = 0, 1, \dots, N - 1$:

1. **Temperature.** Let $n = N - j$ be the number of steps remaining, including step $j$. Compute the linear temperature
   $$t = t_{\min} + (t_{\max} - t_{\min}) \cdot \frac{n}{N}.$$
2. **Predict and scale.** Compute the temperature-scaled logits
   $$z = \frac{\texttt{model.denoise}(x, \texttt{cache}, z^{\text{prev}})}{t},$$
   where $z^{\text{prev}}$ is the scaled logits $z$ from step $j - 1$ of the current canvas. On step $j = 0$ of every canvas, pass no third argument (or pass `None`).
3. **Sample, accept, and renoise.** As in `entropy-bound-denoise-step`, compute $p_i = \operatorname{softmax}(z_i)$ for each position $i$:
   - *Sample a candidate* $y_i$ per position by inverse-CDF sampling:
     $$y_i = \min\{k : C_{i,k} > u \cdot C_{i,V-1}\}, \quad C_{i,k} = \sum_{v=0}^{k} p_{i,v}, \quad u = \texttt{u\_sample}[c, j, i].$$
   - *Accept by entropy bound.* Compute the Shannon entropy $H_i = -\sum_{v=0}^{V-1} p_{i,v} \ln p_{i,v}$, where a probability of 0 contributes 0. Rank positions by ascending $H_i$, breaking ties by smaller index $i$. Position $i$ is accepted if the sum of entropies of all positions ranked strictly before $i$ is at most `entropy_bound`. This matches `ops.eb_accept(z, entropy_bound)`.
   - *Update the canvas.* Set the next step's canvas to $x_i = y_i$ at accepted positions and $x_i = \lfloor \texttt{u\_noise}[c, j, i] \cdot V \rfloor$ (product in float32) at rejected positions.
4. **Record draft.** Set step $j$'s draft to the greedy argmax $d^{(j)}_i = \operatorname{argmax}_v z_{i,v}$ for every position $i$.
5. **Early stop.** Stop denoising this canvas after step $j$ if both conditions hold:
   - *Stable:* $j \ge s$ and $d^{(j)} = d^{(j-1)} = \dots = d^{(j-s)}$, comparing only drafts from the current canvas. When $s = 0$, every step is stable.
   - *Confident:* $\frac{1}{L} \sum_{i=0}^{L-1} H_i < \texttt{confidence}$.

The final draft $d$ of a canvas is always $d^{(j)}$ from the last step that ran, whether the canvas stopped early or reached $N$ steps. It is never the updated canvas $x$ produced by the sampler.

![One call that generates three canvases. Every denoising step reads the prefix from the cache, and the last draft is never encoded.](assets/canvas-loop.svg)

### Return value and rules

- Return a list of Python `int` values containing `prompt` followed by $L$ tokens for each generated canvas.
- Do not modify any input passed to `diffusion_generate`.
- Do not draw random numbers yourself.
- Read `model.eos_id` and `model.pad_id` from `model` rather than hardcoding `1` and `0`, since they vary across test models.
- From `model`, use only `encode`, `denoise`, `vocab_size`, `eos_id`, and `pad_id`. Accessing counters, other settings, private attributes, `with_framework`, or copying `model` is forbidden. Pass only caches returned by `model.encode`.
- You may use `ops.eb_accept`, `ops.softmax`, and `ops.log_softmax`.

### Cost

For a call that generates $g$ canvases, the model's counters must match these exact values:

| counter | value | why |
|---|---|---|
| `encoder_calls` | $g$ | one call for the prompt, plus one for each canvas followed by another canvas |
| `encoder_tokens` | $P + (g - 1) L$ | every prefix token is encoded once, and the final canvas is never encoded |
| `denoise_calls` | exact number of steps defined above | no steps after early stopping, and no repeated calls |

## Shapes

| name | shape | dtype |
|---|---|---|
| `prompt` | list of $P$ ints | Python `int` |
| `u_init` | `(max_canvases, L)` | float32 |
| `u_sample`, `u_noise` | `(max_canvases, max_steps, L)` | float32 |
| canvas, draft | `(L,)` | integer |
| logits, $z$ | `(L, V)` | float32 |
| result | list of $P + g L$ ints | Python `int` |

## Example

```python
# `tensor` stands for your framework's float32 tensor; the prompt is a plain list.
model = BlockDenoiser(fw, seed=18, vocab_size=8, sharpness=3.0, eos_bias=1.0)  # eos_id 1, pad_id 0
cfg = {"canvas_length": 4, "max_canvases": 2, "max_steps": 3, "t_min": 0.4, "t_max": 0.8,
       "entropy_bound": 0.1, "stability": 1, "confidence": 0.005}
u_init = tensor([[0.97, 0.44, 0.20, 0.69], [0.81, 0.29, 0.30, 0.08]])
u_sample = tensor([[[0.40, 0.58, 0.58, 0.34], [0.43, 0.57, 0.32, 0.33], [0.92, 0.92, 0.92, 0.71]],
                   [[0.78, 0.41, 0.94, 0.74], [0.26, 0.70, 0.43, 0.35], [0.21, 0.45, 0.51, 0.42]]])
u_noise = tensor([[[0.71, 0.91, 0.68, 0.94], [0.69, 0.94, 0.43, 0.09], [0.56, 0.71, 0.18, 0.70]],
                  [[0.83, 0.66, 0.17, 0.70], [0.54, 0.16, 0.67, 0.68], [0.83, 0.83, 0.79, 0.05]]])

diffusion_generate(model, [3, 6, 2], cfg, (u_init, u_sample, u_noise))
# [3, 6, 2, 4, 4, 4, 4, 4, 1, 0, 0]
```

| canvas | step | $t$ | canvas in | draft | accepted | canvas out | mean $H$ | stable | confident |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.8000 | `[7, 3, 1, 5]` | `[4, 4, 4, 4]` | 0 and 3 | `[4, 7, 5, 4]` | 0.0967 | no | no |
| 0 | 1 | 0.6667 | `[4, 7, 5, 4]` | `[4, 4, 4, 4]` | all | `[4, 4, 4, 4]` | 0.0000 | yes | yes |
| 1 | 0 | 0.8000 | `[6, 2, 2, 0]` | `[1, 1, 1, 4]` | 1 | `[6, 1, 1, 5]` | 0.5214 | no | no |
| 1 | 1 | 0.6667 | `[6, 1, 1, 5]` | `[1, 1, 1, 4]` | all | `[1, 1, 1, 4]` | 0.1659 | yes | no |
| 1 | 2 | 0.5333 | `[1, 1, 1, 4]` | `[4, 1, 1, 4]` | all | `[1, 1, 1, 4]` | 0.1680 | no | no |

Canvas 0 starts as $\lfloor 8 \cdot (0.97, 0.44, 0.20, 0.69) \rfloor = (7, 3, 1, 5)$ and stops early after step 1 because its draft repeats and its mean entropy is below `0.005`. Its draft `[4, 4, 4, 4]` is appended and encoded into the cache, and canvas 1 starts without self-conditioning. Canvas 1 is stable at step 1 but not confident, so it continues, changes its draft at step 2, and hits `max_steps`. Its final draft is `[4, 1, 1, 4]` even though the sampler's output canvas is `[1, 1, 1, 4]`. The first occurrence of `eos_id` (`1`) at index 1 ends generation, and the two positions after it become `pad_id` (`0`). The final counters read 2 encoder calls, $3 + 4 = 7$ encoder tokens, and 5 denoise calls.

With `max_canvases = 1` and only the first row of each uniform tensor, the same call returns `[3, 6, 2, 4, 4, 4, 4]` after 1 encoder call and 2 denoise calls.

## Constraints

- $1 \le P \le 9$, $1 \le L \le 8$, $V \in \{4, 8, 32\}$, `max_canvases` $\le 5$, `max_steps` $\le 12$, $s \in \{0, 1, 2\}$, $0.2 \le t_{\min} < t_{\max} \le 1.5$, $0.02 \le \texttt{entropy\_bound} \le 1$, and $0.005 \le \texttt{confidence} \le 3$.
- Output tokens are compared exactly against a float64 reference evaluation and against Hugging Face Transformers' `DiffusionGemmaGenerationMixin.generate` driving the same model with the same uniforms.
- Test uniforms are constructed so that float32 rounding never flips any discrete decision. At every step along the correct trajectory:
  - Every `u_sample` threshold $u \cdot C_{i,V-1}$ is at least $10^{-3}$ away from every cumulative sum $C_{i,k}$, and every $u \cdot V$ from `u_init` and `u_noise` is at least $10^{-3}$ away from any integer.
  - Every entropy prefix sum compared against `entropy_bound` is either 0 or at least $10^{-3}$ away from `entropy_bound`, and the entropies of the last accepted position and first rejected position differ by at least $10^{-3}$, so tie-breaking never changes acceptance.
  - The top two entries in every row of $z$ differ by at least $10^{-3}$.
  - The mean entropy differs from `confidence` by at least $\min(10^{-3}, 0.1 \cdot \texttt{confidence})$.
- The three counters (`encoder_calls`, `encoder_tokens`, and `denoise_calls`) are compared exactly.

## Why this matters

Block text-diffusion models are still autoregressive across blocks, and their prefix cache works like a standard KV cache. Whatever tokens you encode into the cache are attended to by every later canvas. At the end of a canvas, three `(L,)` tensors look interchangeable: the sampled candidates, the renoised canvas, and the argmax draft. They match whenever a canvas converges cleanly, but they diverge when a canvas hits `max_steps`. Appending or encoding the wrong tensor passes tests on easy prompts while silently poisoning the cache on harder ones.

The stopping rule controls both quality and latency. Stopping on stability alone emits low-confidence guesses, while failing to stop early forces the model to pay for all 48 denoising passes on every canvas in production.
