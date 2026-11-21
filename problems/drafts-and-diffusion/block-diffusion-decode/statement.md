Implement the generation loop of a block text-diffusion model, the `generate` of
DiffusionGemma for one sequence: the prompt is encoded once into a prefix cache, and the
text then grows one canvas of $L$ tokens at a time, each canvas denoised from noise over
several steps and added to the cache when it is finished. The judge compares exact
tokens and counts the work asked of the encoder and the denoiser.

## Specification

```python
def diffusion_generate(model, prompt, cfg, uniforms): ...  # -> list of ints
```

`model` is a [`BlockDenoiser`](../../../python/src/learnai/kit/diffusion.py) from the
kit, with a vocabulary of $V$ = `model.vocab_size` tokens:

```python
cache = model.encode(tokens)            # a new prefix: tokens is a list of ints or a 1-D tensor
cache = model.encode(tokens, cache)     # the cached prefix followed by tokens
logits = model.denoise(canvas, cache)             # float32 (L, V), for the L tokens after the prefix
logits = model.denoise(canvas, cache, previous)   # the same, conditioned on float32 (L, V)
```

Neither call changes the cache it is given. `canvas` is an integer tensor of shape
`(L,)`. `cfg` is a dict with the keys `canvas_length` ($L$), `max_canvases`,
`max_steps` ($N$), `t_min`, `t_max`, `entropy_bound`, `stability` ($s$), and
`confidence`. Randomness is an input: `uniforms = (u_init, u_sample, u_noise)`.

**The text.** Encode the prompt, `cache = model.encode(prompt)`. Then, for canvas
$c = 0, 1, \dots$, at most `max_canvases` of them:

1. **Noise.** The canvas starts as $x_i = \lfloor \texttt{u\_init}[c, i] \cdot V \rfloor$,
   the product taken in float32.
2. **Denoise** it with the steps below. They end with a *draft* $d$ of $L$ tokens.
3. **Append.** If the draft contains `model.eos_id`, replace every token after its
   first occurrence by `model.pad_id`, append the result to the text, and return. Only
   the draft is searched: the token may occur in the prompt.
4. Otherwise append the draft. If this was canvas `max_canvases - 1`, return.
   Otherwise `cache = model.encode(d, cache)` and continue with the next canvas.

**The steps of one canvas**, $j = 0, 1, \dots$, at most $N$ of them:

1. **Temperature.** $n = N - j$ steps remain, this one included, and
   $t = t_\text{min} + (t_\text{max} - t_\text{min}) \cdot n / N$.
2. **Predict.** $z = \texttt{model.denoise}(x, \texttt{cache}, z^\text{prev}) \,/\, t$,
   where $z^\text{prev}$ is the $z$ of the previous step of this canvas. The first step
   of every canvas passes no third argument.
3. **Sample and accept**, as in `entropy-bound-denoise-step`, from
   $p_i = \operatorname{softmax}(z_i)$. A candidate per position by inverse CDF:
   $y_i = \min\{k : C_{i,k} > u \cdot C_{i,V-1}\}$ with $C_{i,k} = \sum_{v \le k} p_{i,v}$
   and $u$ = `u_sample[c, j, i]`. The entropy $H_i = -\sum_v p_{i,v} \ln p_{i,v}$, where
   a probability of 0 contributes 0. Position $i$ is accepted when the entropies ranked
   strictly before it (ascending, equal ones by index) sum to at most `entropy_bound`;
   this is `ops.eb_accept(z, entropy_bound)`. The next step's canvas is $x_i = y_i$ where
   accepted and $\lfloor \texttt{u\_noise}[c, j, i] \cdot V \rfloor$ elsewhere.
4. **Draft.** $d^{\,j}_i = \operatorname{argmax}_v z_{i,v}$, for every position.
5. **Stop** after this step when both hold:
   - *stable*: $j \ge s$ and $d^{\,j} = d^{\,j-1} = \dots = d^{\,j-s}$, drafts of this
     canvas only. With $s = 0$ every step is stable.
   - *confident*: $\tfrac{1}{L} \sum_i H_i <$ `confidence`.

The draft of a canvas is the $d^{\,j}$ of the last step that ran, whether the canvas
stopped or ran out of steps. It is not the canvas $x$ that the sampler left behind.

![One call that generates three canvases. The encoder reads the prompt and the first two drafts, once each; every step of a canvas reads the whole prefix from the cache; the last draft is appended and never encoded.](assets/canvas-loop.svg)

- `prompt`: a list of $P \ge 1$ ints. `uniforms`: three float32 tensors, $0 \le u < 1$.
- Returns a list of Python ints: the prompt followed by $L$ tokens for every canvas
  generated.
- Nothing passed in is modified.

**Cost.** For a call that generates $g$ canvases, the model's counters must read
exactly:

| counter | value | why |
|---|---|---|
| `encoder_calls` | $g$ | the prompt in one call, then each canvas that another canvas follows |
| `encoder_tokens` | $P + (g - 1)\,L$ | no token is encoded twice, and the last canvas is never read again |
| `denoise_calls` | the number of steps the rules above define | no step after the stop, and none repeated |

Of the model, use `encode`, `denoise`, `vocab_size`, `eos_id`, and `pad_id`, and
nothing else: its counters, its other settings, its private attributes,
`with_framework`, and copies of it are off limits, and it takes only caches that it
returned itself. The two special tokens differ between models (most tests have 1 and
0), so read them. Do not draw random numbers yourself. `ops.eb_accept`, `ops.softmax`,
and `ops.log_softmax` may be used.

## Shapes

| name | shape | dtype |
|---|---|---|
| `prompt` | list of $P$ ints | Python `int` |
| `u_init` | `(max_canvases, L)` | float32 |
| `u_sample`, `u_noise` | `(max_canvases, max_steps, L)` | float32 |
| canvas, draft | `(L,)` | integer |
| logits, $z$ | `(L, V)` | float32 |
| result | list of $P + g\,L$ ints | Python `int` |

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

Canvas 0 starts as $\lfloor 8 \cdot (0.97, 0.44, 0.20, 0.69) \rfloor = (7, 3, 1, 5)$ and
stops after its second step. Its draft `[4, 4, 4, 4]` is appended and encoded, and
canvas 1 starts without self-conditioning. Canvas 1 is stable at step 1 but not
confident, so it goes on, changes its draft, and runs out of steps. Its last draft is
`[4, 1, 1, 4]` although the sampler's canvas is `[1, 1, 1, 4]`; the first token 1 ends
the text and the two tokens after it become padding. The counters read 2 encoder calls,
$3 + 4 = 7$ encoder tokens, and 5 denoise calls.

With `max_canvases = 1` and the first row of each uniform tensor, the same call returns
`[3, 6, 2, 4, 4, 4, 4]` after 1 encoder call and 2 denoise calls.

## Constraints

- $1 \le P \le 9$, $1 \le L \le 8$, $V \in \{4, 8, 32\}$, at most 5 canvases and 12 steps,
  $s \in \{0, 1, 2\}$, $0.2 \le t_\text{min} < t_\text{max} \le 1.5$,
  $0.02 \le$ `entropy_bound` $\le 1$, $0.005 \le$ `confidence` $\le 3$.
- Tokens are compared exactly, with the loop evaluated in float64 and with the output of
  `DiffusionGemmaGenerationMixin.generate` of Hugging Face Transformers driving the same
  model with the same uniforms. The uniforms are built so that float32 rounding decides
  nothing. At every step the correct loop takes:
  - every `u_sample` threshold $u \cdot C_{i,V-1}$ is at least $10^{-3}$ away from every
    cumulative sum, and every $u \cdot V$ of `u_init` and `u_noise` at least $10^{-3}$
    away from an integer;
  - every sum of entropies that is compared with `entropy_bound` is 0 or at least
    $10^{-3}$ away from it, and the entropies of the last accepted and the first rejected
    position are at least $10^{-3}$ apart, so the tie rule never decides;
  - the two largest entries of every row of $z$ are at least $10^{-3}$ apart;
  - the mean entropy is away from `confidence` by at least $10^{-3}$, or by a tenth of
    `confidence` where that is smaller.
- The three counters are compared exactly (see Cost).

## Why this matters

A text-diffusion model is still autoregressive, over blocks instead of tokens, and its
prefix cache is an ordinary one: what goes into it is read by every later canvas. The
loop has three tensors that look alike at the end of a canvas, the sampled canvas, the
renoised canvas, and the argmax draft. They are equal when a canvas converges and
different when it runs out of steps, so appending or encoding the wrong one passes every
test on easy prompts and corrupts the hard ones. The stopping rule decides the cost: a
canvas that stops on stability alone ships a guess, and one that never stops early pays
for every one of its 48 passes on every canvas.
