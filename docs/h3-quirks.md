# The MiniMax-H3 contract

Everything on this page is a place where H3 differs from what training code usually assumes. Each one
is silent when you get it wrong: no error, no shape mismatch, often a loss curve that looks fine.

## 1. Time runs opposite to sigma

The transformer's time input is `t = 1 - sigma`. **`t = 1` is clean data, `t = 0` is pure noise.** The
scheduler's forward process is `x_t = t * x0 + (1 - t) * noise`, i.e. the usual
`x_t = (1 - sigma) * x0 + sigma * noise`.

Feed sigma where t belongs and every sample is labelled with the opposite noise level. The model still
trains - to invert its own schedule.

## 2. The prediction is a data-ward velocity

H3 predicts `v = x0 - eps`, and the scheduler denoises with `x0 = x_t + (1 - t) * v` - note the plus.
Most flow-matching code regresses `eps - x0`. That sign error trains the model to move *away* from the
data at every step.

```python
target = clean - noise           # correct
target = noise - clean           # trains the model backwards
```

`tests/test_flow_matching.py::test_reconstruction_identity_holds` pins the identity.

## 3. Video and audio are noised at different sigmas

H3 carries two schedulers, `shift = 12.0` for video and `shift = 3.0` for audio, and both step in
lockstep off a shared step index. Training must reproduce that pairing: draw **one** uniform `u` and
map it through both curves.

```python
sigma = shift * u / (1 + (shift - 1) * u)
```

Two independent draws produce (video, audio) noise pairings the model never sees at inference. This is
the single most likely place to break joint audio-video training while everything still "works".

## 4. Silent clips must not train the audio head

A clip with no soundtrack still occupies audio rows in the packed sequence - the geometry is fixed by
the video length. Regressing those zero latents teaches the audio head to predict silence (in latent
space, to predict *noise*).

The fix is to weight the audio term to 0, not to drop it. Dropping the term removes the audio head
from the autograd graph on some ranks and not others, and DDP/ZeRO then disagree about which gradients
exist - which shows up as a hang, usually minutes later.

## 5. Conditioning rows are inputs, not targets

The packed sequence is:

```
[ text | conditioning blocks | target audio | target video ]
```

Conditioning rows (keyframes, IC-LoRA references) take part in self-attention in both directions, but:

* **visual** conditioning is noise-augmented to `t = 0.999` (sigma = 0.001) - a whisper of noise, not
  clean data, because that is what inference feeds;
* **reference audio** is passed through completely clean at `t = 1.0` - the inference path applies no
  augmentation to it;
* neither ever contributes to the loss.

## 6. Conditioning latents are encoded differently from targets

Targets take the posterior **mode**. Conditioning reproduces the inference recipe exactly:

```python
posterior.sample(generator=torch.Generator().manual_seed(42))   # fixed seed, not the request seed
latents = latents.to(torch.float16).float()                     # ~11 bits thrown away, on purpose
latents = (latents - latents_mean) / latents_std
```

The float16 round-trip is part of the contract. The released model was conditioned on latents that had
been through it; conditioning it on full-precision latents at training time is a train/inference
mismatch in the one signal you are trying to teach it to follow.

Images go through the spatial encoder (`_encode_clip`); videos go through the temporal chunking
(`_encode`) that turns `17n + 5` frames into `5n + 2` latent frames.

## 7. Vision blocks live in the prompt, tagged as video

Keyframes and references are not only latent rows. They also appear in the conditioner's *presentation*
of the request: each prepends a `"<Picture i>: "` (or `"<Video k>: "` / `"<Audio j>: "`) label and, for
visual media, a vision block of `<|vision_start|> … <|image_pad|> … <|vision_end|>` tokens.

The rows of a vision block are tagged **video** (`0`), not text (`1`), and the transformer's AdaLN
modulation keys off that tag. So the cached prompt embedding must be produced by running Qwen3-VL over
the *whole presentation*, and the per-row tags must be cached alongside it. `process_dataset.py` does
both; the tags travel in `conditions/<id>.safetensors` as `text_token_tags`.

## 8. Geometry

| constraint | value |
|---|---|
| frame rate | exactly 24.000 fps |
| frame count | `17n + 5` → 22, 39, 56, 73, 90, 107, 124, … |
| latent frames | `5n + 2` |
| height / width | divisible by 32 (VAE 16x, patch 2x) |
| duration | **5-15 s** to generate (the pinned pipeline rejects less; MiniMax's own README says 4, the code says 5). Training packs any `17n+5` length, but a clip under 5 s is out of the distribution the model generates. |
| audio | 32 kHz stereo, 40 Hz latent grid, 800 samples per latent |
| audio rows | channel-major: `[ch0 × N, ch1 × N]`, **not** interleaved |

A 25 fps clip used unchanged is a clip in 4% slow motion, and systematic slow motion is one of the
first things a LoRA learns. Resample, don't hope.

## 9. No classifier-free guidance

The released checkpoints are guidance-distilled: guidance is baked into the weights. There is no
negative prompt, no CFG scale, and every denoising step is a single forward pass. Config keys for
those knobs would do nothing, so this trainer does not have them.

## 10. Distributed landmines

* **Activation checkpointing under ZeRO-3** must use `deepspeed.checkpointing.checkpoint`. Torch's
  implementation (reentrant or not) recomputes against parameters DeepSpeed has already
  re-partitioned, and sees shape-`[0]` tensors.
* **Never put logic between `backward()` and `step()`.** It breaks ZeRO-3's gradient-accumulation
  bookkeeping (`KeyError: averaged_gradients`). Sequence-length decisions belong *before* the forward.
* **Skip decisions must be all-reduced.** A rank that skips a step alone leaves the others waiting on
  a collective forever.
* **Save only trainable tensors.** A full H3 state dict is ~66GB; writing it from rank 0 stalls long
  enough to trip the NCCL watchdog.
* **Resume before sharding.** After ZeRO-3 partitions parameters, `load_state_dict` sees shape-`[0]`
  shards and loads nothing, silently.
* **Checkpoint before validating.** A validation forward leaves ZeRO-3 prefetch in flight, and a
  parameter gather immediately after fails with "Cannot partition a param in flight".
* **`expandable_segments:True`** must be set before torch is imported.


## Measurements from prior work

Numbers reported by [MiniMax-H3-FineTuning](https://github.com/IAmIronMan42/MiniMax-H3-FineTuning),
worth having in one place. Measured on 8×A800-80GB, not reproduced here.

| measurement | value |
|---|---|
| Sequence ceiling, full attention + LoRA + ZeRO-3 | **≈70k tokens** (65k ⇒ 76GB steady; 76k OOMs) |
| Throughput, 65k-token 30s sequences, LoRA | ~7.5-8 min/step |
| Throughput, heads-only at 33k tokens | ~53 s/step |
| Largest reported run | LoRA over 2,000 ~30s clips at 448×768 |

**Why the conventions on this page are not pedantry.** With the timestep direction and velocity sign
inverted, they report a heads-only run whose loss *rises* - 7.2 → 9.5 over 10 steps. Corrected, the
same setup sits at 0.3-1.0, stable over 1000 steps. That is the clearest available evidence that these
details decide whether a run trains at all.

**Sparse attention** - used in H3's final training stage - is **not released**. Training therefore runs
full attention, which is what sets the token ceiling above, and it is unlikely to improve.

Two knobs enforce the budget without re-encoding a cache: `H3_MAX_LF` (truncate cached samples to *n*
latent frames at load time) and a pre-flight all-reduced skip gate, which this trainer implements as
`optimization.max_seq_tokens`.
