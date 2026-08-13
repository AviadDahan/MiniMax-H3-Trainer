# Troubleshooting

Failures on the left are the ones that cost real time here; each was hit and fixed while building
this trainer.

## Setup

**`ImportError: MiniMax-H3 classes are not in any released diffusers wheel`**
Install the pinned commit: `pip install --no-deps
'git+https://github.com/huggingface/diffusers.git@abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc'`, or run
`scripts/install_env.sh`.

**`'Qwen3VLProcessor' object has no attribute 'create_mm_token_type_ids'`**
`transformers` is too old. H3's conditioner needs Qwen3-VL's per-token modality ids, and neither the
processor helper nor the model argument exists in 4.57.x. Install `transformers >= 5.15`. (The public
H3 reference trainer pins 4.57.3; that pin cannot work with the diffusers integration.)

**`TorchCodec is required for load_with_torchcodec`**
Recent `torchaudio.load` delegates to an optional codec backend. This trainer avoids it: ffmpeg does
the decode and the resulting 16-bit PCM wav is read with the standard library. If you hit this in
your own code, do the same or install `torchcodec`.

## Memory

**OOM loading the text encoder.** Qwen3-VL-32B is ~64GB in bf16 and does not fit on a 48GB card.
Preprocessing spreads it with `device_map="auto"`; keep `--text-device-map auto` (the default).

**OOM in `deepspeed_zero3` before training starts.** ZeRO-3 partitions *after* each rank has
constructed the model, so every rank needs to hold 66GB first. That works on 80GB cards and not on
48GB ones. On smaller cards use `strategy: model_parallel` (one process, blocks split across GPUs,
full bf16) or `strategy: ddp` with `quantization: nf4-bnb`.

**OOM mid-run at a particular sample.** Lower `optimization.max_seq_tokens` - over-budget samples are
then skipped before the forward pass instead of exploding during it. Check the `skipped_long` counter
in the logs so you know how much of your data is being dropped.

**Allocator fragmentation.** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` must be set *before*
torch is imported. `scripts/env.sh` does it; the scripts also set it defensively at the top.

**`nvidia-smi` shows memory still held after preprocessing.** A model placed with `device_map` stays
alive through accelerate's dispatch hooks even after you drop your reference. `H3Encoders.unload()`
removes the hooks and collects; do the same in your own code.

## Distributed

**DeepSpeed ignores `CUDA_VISIBLE_DEVICES`.** Passing `--num_gpus`/`--include` makes the launcher
override it - `deepspeed --num_gpus 4` with `CUDA_VISIBLE_DEVICES=4,5,6,7` runs on GPUs **0-3**. Use
`deepspeed --include localhost:4,5,6,7` instead.

**Training hangs at a collective.** Something diverged between ranks. The usual causes: a skip
decision made per-rank instead of all-reduced, a validation loop with different sample counts per
rank, or a loss term that exists on some ranks and not others. All three are handled here (the
sequence gate is all-reduced, validation uses the global minimum count, and the audio term stays in
the graph at weight 0) - if you add code, keep those invariants.

**`KeyError: averaged_gradients`.** Something ran between `backward()` and `step()` under ZeRO-3.
Move it before the forward pass.

**`Cannot partition a param in flight` when saving.** A checkpoint gather ran right after a
validation forward, while ZeRO-3 prefetch was still outstanding. Checkpoint *before* validating.

**Checkpoint saved but resume loads nothing.** Weights must be loaded while parameters are still full
tensors, i.e. before `deepspeed.initialize()`. After partitioning, `load_state_dict` sees shape-`[0]`
shards and silently succeeds. This trainer also treats unexpected keys as fatal, so a checkpoint from
a different adapter configuration fails loudly instead of loading nothing.

## Training looks wrong

**`lora.target_modules entries matched no Linear layer`.** Working as intended. PEFT drops unmatched
targets silently, so a name that matches nothing shrinks your adapter without telling you. Use
`to_q`, `to_k`, `to_v`, `to_out.0` (and `ff.net.0.proj`, `ff.net.2` for more capacity) - the names in
the diffusers checkpoint. `to_qkv`, `qkv_proj`, `linear_1`, `linear_2` come from the original MiniMax
packaging and match nothing here.

**Total loss falls, `loss_audio` is flat.** The audio branch is not learning. Check `audio_weight` in
the logs: if it is 0, your clips have no usable audio track and the audio term is being weighted out
by design. Re-encode with audio.

**Loss is noisy and jumps by 10x between steps.** Expected. Loss is strongly sigma-dependent, and each
step samples a different noise level. Watch the per-sigma validation curves
(`val/loss_video_u0.3`, `u0.6`, `u0.9`) instead - those are seeded and comparable across steps.

**Training runs fine, samples are garbage.** In order of likelihood: (1) the encoding recipe is wrong
- run `process_dataset.py --decode 3` and *watch* the round-trips; (2) a numeric convention is
inverted - run `pytest tests/test_flow_matching.py`; (3) the base model is quantized too aggressively
- see below.

**An entire epoch executed zero steps.** Every sample exceeded `max_seq_tokens`. Lower the resolution
bucket or raise the budget; the trainer raises rather than spinning.

## Generation

**Output is coloured static.** Decode a tensor of random latents and compare: if it looks the same,
the transformer produced noise rather than the VAE mis-decoding it. The cause here was 4-bit
quantization - NF4 fits the model on one 48GB card, but quantizing H3's AdaLN modulation branches to
4 bits destroys generation. Use `--placement shard` (bf16 across GPUs) for anything you intend to
look at, and keep 4-bit for memory-bound training only.

**`Expected all tensors to be on the same device … index is on cuda:0, other tensors on cuda:7`.**
A plain `device_map="auto"` split the transformer so the output projection landed on a different GPU
than the packed layout's index vectors. Use `--placement shard`, which distributes only
`transformer_blocks.*` and pins every index-consuming module to one device.

**`NoneType object is not callable` inside the denoise loop.** The transformer was registered under
the wrong component name. The Ref2VA blocks call it `transformer_ref`, not `transformer`, and
`update_components` accepts an unknown name without complaint.

**`num_frames … must be between 120 and 360`.** H3 generates 5-15s only. Valid counts: 124, 141, 158,
175, 192, 209, 226, 243, 260, 277, 294, 311, 328, 345. Training accepts shorter clips; generation
does not.

**`Required input 'references' is missing`.** The Ref2VA setup block wants the raw reference
descriptors, not just prepared ones.

**Generated audio is silent or an octave too high.** Muxing: H3's audio is planar `(2, N)` and packed
s16 wants it interleaved. Flattening the planar tensor row-major concatenates the channels, which
plays back at double speed and an octave high.
