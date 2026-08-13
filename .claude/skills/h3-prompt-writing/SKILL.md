---
name: h3-prompt-writing
description: Write prompts and captions for MiniMax-H3 training with this trainer. Use when captioning a dataset, choosing where a LoRA trigger goes, describing the soundtrack that trains jointly with the picture, writing validation prompts, or working out why an adapter memorized the room instead of the subject.
---

# Prompt writing for H3 training

Written for **training-time** captioning with this trainer. For the full inference prompt
specification - the structured field format MiniMax trained the released model on - their own guide is
the authority and ships in their repository: https://github.com/MiniMax-AI/MiniMax-H3 under
`.claude/skills/h3-prompt-writing`. It is not reproduced here.

## What the trainer actually does with a caption

The caption is encoded once, offline, by Qwen3-VL-32B at layer 50, and the resulting rows are packed
at the **front** of every sequence. Two consequences:

* The pipeline passes the prompt through **verbatim** - no chat template, no special tokens. Whatever
  you write is what the model conditions on.
* Caption length changes the row count, which is why samples only batch together when their captions
  match in length. `text_rows` is recorded per sample in `index.json`.

Re-caption and you must re-run the text pass; the embedding is cached, not the string.

## Describe the sound. It is half the model.

H3 generates video and audio in one pass, and the audio branch trains from your caption too. A
caption that describes only the picture leaves the audio head learning from nothing, which shows up
as a flat `loss_audio` under a healthy-looking total.

Say what the scene *sounds* like: ambience, the physical sounds of the action, and - if anyone speaks
- the voice and the words.

## Train and generate in the same register

The single most useful habit: **write captions in the form you intend to prompt in.** An adapter
trained on terse captions and prompted with three paragraphs is being asked to generalize across a
distribution shift you introduced for no reason.

This cuts both ways. If your inference prompts will use MiniMax's structured fields
(`integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music` - in that order),
caption in that shape too. If they will be flowing prose, caption in flowing prose. Pick one.

## The trigger goes in exactly one place

Either bake it into every caption, or pass `--lora-trigger` at preprocessing. **Never both** -
duplicating it degrades prompt adherence.

Use a token the model has no prior for. A real name drags in everything the base model already
associates with it; something like `OHWXMIRA` starts empty.

## Vary everything except the thing you are teaching

The anti-memorization rule, and the one most often skipped. An adapter learns whatever is *constant*
across the dataset, not what you intended it to learn.

Our 36-clip character run makes the failure concrete: identity held across unseen scenes, and the
adapter also learned the anchor's orange sweater, because every clip had it. Backgrounds, framing,
lighting, action and wardrobe all have to move.

A worked example from that dataset:

```
[VISUAL] OHWXMIRA, a medium close-up of a woman seated at a kitchen table in morning light.
[SPEECH] OHWXMIRA speaks in a warm, slightly husky mid-range voice: "I keep meaning to write this down."
[SOUNDS] cutlery clinking faintly and a kettle in the background.
```

The tags are ours, not a model requirement - they exist so a human can see at a glance that the sound
was actually described.

## When the reference carries the content

For structural IC-LoRA the caption should describe *what the video is*, not repeat what the reference
already encodes. The skeleton says where the limbs are; the caption says it is a person dancing. Our
pose dataset uses one caption for every pair, deliberately:

```
a person dancing, full body in frame, following the motion of the reference skeleton
```

## Validation prompts

Put at least one **untriggered** prompt in `validation.samples`. It is the check that catches
prompt-adherence collapse - an adapter that rewrites every unrelated prompt is broken, and nothing in
the loss will tell you.

## Two constraints worth knowing before you write

* **No negative prompt.** The checkpoints are guidance-distilled: there is no CFG scale and no
  unconditional branch, so "not blurry, no watermark" does nothing. Describe what you want.
* **5-15 seconds.** H3 generates nothing shorter or longer, so a caption describing a 30-second
  sequence of events cannot be satisfied.
