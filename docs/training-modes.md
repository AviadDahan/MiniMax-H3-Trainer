# Training modes

There is one strategy, `flexible`. Every mode below is that strategy with different flags — no mode
has its own code path. Adding a mode should mean writing a YAML file.

Two questions are asked per modality:

1. **Is it generated?** `is_generated: true` → the modality is noised, predicted, and in the loss.
   `false` → it is packed clean (sigma = 0, t = 1) as frozen conditioning and excluded from the loss.
2. **What conditions ride in front of it?** Keyframes (`first_frame`, `last_frame`) or in-context
   `reference` blocks.

```
[ text | conditioning blocks | target audio | target video ]
```

## Quick reference

| mode | config | video | audio | conditions | variant |
|---|---|---|---|---|---|
| text → video+audio | [`t2va_lora.yaml`](../configs/t2va_lora.yaml) | generated | generated | — | fl2va |
| text → video+audio (48GB) | [`t2va_lora_low_vram.yaml`](../configs/t2va_lora_low_vram.yaml) | generated | generated | — | fl2va |
| image → video+audio | [`i2v_lora.yaml`](../configs/i2v_lora.yaml) | generated | generated | `first_frame` | fl2va |
| first+last → video | — | generated | generated | `first_frame`, `last_frame` | fl2va |
| video → audio | [`v2a_lora.yaml`](../configs/v2a_lora.yaml) | **frozen** | generated | — | fl2va |
| audio → video | — | generated | **frozen** | — | fl2va |
| IC-LoRA (reference) | [`ref2va_ic_lora.yaml`](../configs/ref2va_ic_lora.yaml) | generated | generated | `reference` | **ref2va** |

## Text to video + audio

The baseline. Both modalities are generated, so the adapter learns the joint distribution H3 actually
produces — including what the scene *sounds* like.

```yaml
training_strategy:
  name: flexible
  video: { is_generated: true, latents_dir: latents }
  audio: { is_generated: true, latents_dir: audio_latents }
```

## Image to video (first-frame conditioning)

```yaml
video:
  is_generated: true
  conditions:
    - type: first_frame
      latents_dir: first_frame_latents
      probability: 0.9
```

Preprocess with `--keyframes first_frame`. That does two things: encodes the keyframe with the
inference conditioning recipe (sampled posterior, seed 42, float16 round-trip), and puts its
`"<Picture 1>: "` label plus vision block into the prompt presentation, tagged as video rows.

`probability` below 1.0 leaves a slice of unconditioned steps so the adapter does not forget how to
generate from text alone. `last_frame` works identically and can be combined.

## Video to audio / audio to video

Flip `is_generated` on the modality you want frozen:

```yaml
video: { is_generated: false, latents_dir: latents }     # clean, no loss
audio: { is_generated: true,  latents_dir: audio_latents }
```

The frozen modality is packed at sigma = 0 and masked out of the loss, so the model learns to produce
one modality *given* the other. Useful for foley, dialogue replacement, or driving video from a voice
track.

## IC-LoRA (in-context reference conditioning)

This is the mode nothing else trains today. It requires `model.variant: ref2va` — the FL2VA
transformer has no reference rows in its layout.

```yaml
model:
  variant: ref2va
training_strategy:
  video:
    is_generated: true
    conditions:
      - type: reference
        modality: video          # image | video | audio
        latents_dir: reference_latents
        probability: 0.9
```

Preprocess with `--references` and a `reference_image` / `reference_video` / `reference_audio` column.

What happens at training time:

* reference latents are packed as blocks between the text and the targets, in request order;
* a video reference's soundtrack rows are packed immediately *before* its video rows and share their
  rotary origin, exactly as generated audio and video do;
* visual reference rows are noise-augmented to `t = 0.999`, reference audio rows are passed clean at
  `t = 1.0`;
* reference rows attend bidirectionally with everything else and are masked out of the loss;
* the prompt presentation includes each reference's label and vision block, with those rows tagged as
  video.

H3 accepts up to 9 image, 3 video and 3 audio references per request. An audio reference cannot stand
alone — it needs at least one visual reference or the prompt to anchor the generation.

**Reference dropout matters here.** With `probability: 1.0` the adapter only ever sees a reference and
loses the unconditioned path; 0.8–0.9 keeps both alive. The exception is structural conditioning,
below.

> **Known gap in dropout.** Dropping removes the reference *rows*, but not the reference from the
> *prompt*: one embedding is cached per sample and it is built with the references present, labels and
> vision blocks included. A dropout step therefore describes a reference whose rows are missing —
> a state inference cannot produce. The fix is a second cached embedding built without the
> presentation; until then, `probability: 1.0` is exact and lower values are approximate.

## Structural control: pose, depth, edges

H3 ships no ControlNet and no structural input of any kind. IC-LoRA is the route to one: render the
control signal as a video, pack it as an in-context reference, and let the adapter learn that the
target should follow it. Nothing in the packing is special-cased for this — a skeleton video is a
video reference like any other. What changes is how you configure it.

```yaml
conditions:
  - type: reference
    modality: video
    latents_dir: reference_latents
    probability: 1.0        # not 0.9 -- see below
```

* **`probability: 1.0`.** An identity reference is a hint, so dropping it occasionally is healthy. A
  structural reference *is the instruction*: steps without one teach the adapter to invent motion,
  which is the exact failure this mode exists to prevent.
* **Bucket for the subject.** Square crops of full-body footage cut the legs off, and an adapter that
  never sees ankles cannot place feet. A portrait bucket is the point; the worked example uses
  `320x576x124`.
* **Budget for two clips, and know which canvas you are on.** A reference costs as many rows as the
  clip it is — 448×768×124 with a matching reference measured 27,364 rows at ~82 s/step on four
  A6000s, against 9,970 rows at ~19 s for an unconditioned 512×512 clip. Attention over the packed
  sequence is quadratic, so this dominates everything. Note also that `--reference-canvas native`
  (the default, and what inference does) puts the reference on its *own* 768-short-edge canvas
  regardless of your bucket, which is tens of thousands of rows on its own; `--reference-canvas
  target` trades that fidelity for a run you can afford. Set `optimization.max_seq_tokens` above the
  real length or samples are silently skipped.
* **Alignment is the whole signal.** The control video must be frame-aligned with its target at
  exactly 24.000 fps. A one-frame drift teaches a one-frame lag.

`scripts/extract_pose.py` produces such pairs from ordinary footage, rendering MediaPipe skeletons
with colour-coded bones (limbs must be distinguishable, which is why OpenPose renderings are coloured
rather than white) and rejecting clips that do not show a full body for most of their length. The same
shape works for depth or edge maps: swap the renderer, keep the manifest columns.
`configs/pose_ic_lora.yaml` is the worked example.

## Running them on the hardware you have

Every mode above is orthogonal to `acceleration.strategy`. On 80GB cards, `deepspeed_zero3` gives
data parallelism with full precision. On 48GB cards it cannot start -- each rank builds the whole
66GB model before ZeRO partitions it -- so use `model_parallel`, which splits one bf16 copy across
the GPUs in a single process. See the [configuration reference](configuration-reference.md#acceleration).

## Batching

H3's batch axis is a pure replication axis over *one shared layout*: `position_ids`, `token_tags` and
the index vectors are shared across the batch. Only samples with identical row counts — including
caption length — can share a micro-batch, which natural captions rarely do.

`BucketBatchSampler` groups samples by `(video_rows, audio_rows, text_rows, has_audio)` and shuffles
within and across buckets each epoch. In practice `batch_size: 1` with
`gradient_accumulation_steps: N` is the reliable way to get an effective batch of N; there is no
padding path. Not because padding is unsupported -- rows tagged `-1` form their own attention
document, which is how the reference implementation pads to a multiple of 64 for FlashAttention --
but because padding cannot fix the actual constraint. The structural tensors describe *one* layout
shared by the whole batch, so two differently shaped samples would need different `token_tags`, and
no amount of padding gives them the same ones.
