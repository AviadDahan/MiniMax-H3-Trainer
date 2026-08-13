"""The h3-trainer configuration schema.

The section layout deliberately mirrors LTX-2's ``ltx-trainer`` -- ``model``,
``lora``, ``training_strategy``, ``optimization``, ``acceleration``, ``data``,
``validation``, ``checkpoints``, ``flow_matching``, ``hub``, ``wandb`` -- so a
config from either trainer reads the same way. The H3-specific parts are the
``model.variant`` switch (FL2VA vs Ref2VA transformer), the two flow-matching
shifts, and the sequence-length budget.

``extra="forbid"`` everywhere: a typo'd key should fail at load, not be silently
ignored for six hours of training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from h3_trainer.constants import (
    DEFAULT_AUDIO_SHIFT,
    DEFAULT_VIDEO_SHIFT,
    Geometry,
)

#: LoRA targets for the H3 transformer block, taken from the actual parameter
#: names in the diffusers checkpoint (``transformer_blocks.N.attn.to_q`` ...).
#:
#: The original MiniMax packaging ships a *fused* ``qkv_proj``, and community H3
#: LoRAs are published against that layout -- but the diffusers conversion splits
#: it into to_q/to_k/to_v, which is what PEFT sees here. ``lora.py`` re-fuses on
#: export so the adapter still loads in ComfyUI.
DEFAULT_LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]
#: Adding the SwiGLU feed-forward roughly triples adapter capacity.
FFN_LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]


class ConfigBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Conditioning
# =============================================================================


class FirstFrameConditionConfig(ConfigBaseModel):
    """Keyframe conditioning on the first frame (the I2V case).

    Packed as a conditioning block ahead of the target rows, pinned near t=1 and
    excluded from the loss.
    """

    type: Literal["first_frame"] = "first_frame"
    latents_dir: str = "first_frame_latents"
    probability: float = Field(default=1.0, ge=0.0, le=1.0)


class LastFrameConditionConfig(ConfigBaseModel):
    """Keyframe conditioning on the last frame (FL2VA's second anchor)."""

    type: Literal["last_frame"] = "last_frame"
    latents_dir: str = "last_frame_latents"
    probability: float = Field(default=1.0, ge=0.0, le=1.0)


class ReferenceConditionConfig(ConfigBaseModel):
    """In-context reference conditioning -- this is what makes a LoRA an IC-LoRA.

    Pre-encoded reference latents are concatenated into the packed sequence ahead
    of the targets. Reference rows take part in self-attention in both directions,
    are pinned at the conditioning timestep, and carry no loss. Requires the
    Ref2VA transformer (``model.variant: ref2va``).
    """

    type: Literal["reference"] = "reference"
    modality: Literal["image", "video", "audio"] = "video"
    latents_dir: str = "reference_latents"
    probability: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Per-sample probability of applying the reference. Below 1.0 the adapter also "
        "learns to work unconditioned, which keeps plain text-to-video from collapsing.",
    )


ConditionConfig = Annotated[
    FirstFrameConditionConfig | LastFrameConditionConfig | ReferenceConditionConfig,
    Field(discriminator="type"),
]


class ModalityConfig(ConfigBaseModel):
    """How one modality (video or audio) participates in training."""

    is_generated: bool = Field(
        default=True,
        description="True: the model denoises this modality and it carries loss. False: it is packed "
        "as clean conditioning and excluded from the loss (this is how a2v / v2a are expressed).",
    )
    latents_dir: str = Field(default="latents")
    conditions: list[ConditionConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _reject_keyframes_on_audio(self) -> ModalityConfig:
        return self


class FlexibleStrategyConfig(ConfigBaseModel):
    """The single conditioning framework, driven entirely by configuration.

    t2va, fl2va/i2v, a2v, v2a and every IC-LoRA variant are combinations of
    ``is_generated`` and ``conditions`` -- there is no second strategy class.
    """

    name: Literal["flexible"] = "flexible"
    video: ModalityConfig = Field(default_factory=lambda: ModalityConfig(latents_dir="latents"))
    audio: ModalityConfig = Field(default_factory=lambda: ModalityConfig(latents_dir="audio_latents"))

    @model_validator(mode="after")
    def _require_a_target(self) -> FlexibleStrategyConfig:
        if not (self.video.is_generated or self.audio.is_generated):
            raise ValueError(
                "At least one of training_strategy.video.is_generated / .audio.is_generated must be "
                "true -- otherwise nothing is being trained."
            )
        for condition in self.audio.conditions:
            if condition.type in ("first_frame", "last_frame"):
                raise ValueError(f"'{condition.type}' conditioning is a video concept; it cannot target audio.")
        return self

    def data_sources(self) -> dict[str, str]:
        """Directory (relative to ``data.preprocessed_data_root``) for every tensor we load."""
        sources = {"video": self.video.latents_dir, "audio": self.audio.latents_dir, "text": "conditions"}
        for modality in (self.video, self.audio):
            for condition in modality.conditions:
                sources[condition.type] = condition.latents_dir
        return sources

    @property
    def uses_references(self) -> bool:
        return any(c.type == "reference" for c in (*self.video.conditions, *self.audio.conditions))

    @property
    def keyframe_anchors(self) -> tuple[str, ...]:
        anchors = []
        for condition in self.video.conditions:
            if condition.type == "first_frame":
                anchors.append("first")
            elif condition.type == "last_frame":
                anchors.append("last")
        return tuple(anchors)


# =============================================================================
# Model / LoRA
# =============================================================================


class ModelConfig(ConfigBaseModel):
    model_path: Path = Field(
        ...,
        description="Local MiniMax-H3 directory in the diffusers layout (transformer/, transformer_ref/, "
        "text_encoder/, vae/, audio_vae/, ...).",
    )
    variant: Literal["fl2va", "ref2va"] = Field(
        default="fl2va",
        description="fl2va loads subfolder 'transformer' (text / first / first+last frame conditioning); "
        "ref2va loads 'transformer_ref' (omni-reference: up to 9 images, 3 videos, 3 audio clips).",
    )
    training_mode: Literal["lora", "full", "heads"] = Field(
        default="lora",
        description="lora: PEFT adapters. full: all 33B parameters (needs deepspeed_zero3). "
        "heads: proj_out + audio_proj_out only (~1M params) -- a pipeline smoke test, not a real fine-tune.",
    )
    load_checkpoint: Path | None = Field(
        default=None,
        description="Checkpoint file, or a directory from which the latest checkpoint is taken.",
    )

    @field_validator("model_path")
    @classmethod
    def _must_exist(cls, value: Path) -> Path:
        if not Path(value).exists():
            raise ValueError(f"model_path does not exist: {value}")
        return Path(value)

    @property
    def transformer_subfolder(self) -> str:
        return "transformer_ref" if self.variant == "ref2va" else "transformer"


class LoraConfig(ConfigBaseModel):
    rank: int = Field(default=16, ge=1)
    alpha: int = Field(default=16, ge=1)
    dropout: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Keep at 0 when gradient checkpointing is on: the recomputed forward must match "
        "the original one, and dropout makes it stochastic.",
    )
    target_modules: list[str] = Field(default_factory=lambda: list(DEFAULT_LORA_TARGET_MODULES))
    init_lora_weights: Literal["gaussian", "true"] = "gaussian"

    @field_validator("target_modules")
    @classmethod
    def _reject_names_that_match_nothing(cls, value: list[str]) -> list[str]:
        # These come from the original MiniMax packaging (fused qkv) or from LTX-2.
        # PEFT only raises when *no* target matches anything, so a name that matches
        # nothing is silently dropped -- the reference H3 trainer targets "to_qkv"
        # and as a result never puts an adapter on Q/K/V at all. Fail here instead.
        wrong_names = {
            "to_qkv": "the diffusers checkpoint splits QKV; use to_q / to_k / to_v",
            "qkv_proj": "the diffusers checkpoint splits QKV; use to_q / to_k / to_v",
            "linear_1": "matches only the time embedder here; the FFN is ff.net.0.proj / ff.net.2",
            "linear_2": "matches only the time embedder here; the FFN is ff.net.0.proj / ff.net.2",
            "attn1.to_q": "LTX-2 naming; H3 blocks have a single 'attn'",
            "attn2.to_q": "LTX-2 naming; H3 blocks have a single 'attn'",
        }
        offenders = {name: why for name, why in wrong_names.items() if name in value}
        if offenders:
            details = "; ".join(f"{name!r} -- {why}" for name, why in offenders.items())
            raise ValueError(
                f"lora.target_modules contains names that do not exist in the H3 diffusers "
                f"transformer: {details}. A sane default is {DEFAULT_LORA_TARGET_MODULES}."
            )
        return value


# =============================================================================
# Optimization / acceleration / data
# =============================================================================


class OptimizationConfig(ConfigBaseModel):
    learning_rate: float = Field(default=1e-4, gt=0)
    steps: int = Field(default=2000, ge=1)
    batch_size: int = Field(
        default=1,
        ge=1,
        description="Packed sequences per optimizer micro-step. Only samples with an identical packed "
        "layout can share a batch (H3's batch axis is a pure replication axis over one shared layout), "
        "so anything above 1 needs a dataset with uniform geometry AND uniform caption length. "
        "Prefer gradient_accumulation_steps.",
    )
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    max_grad_norm: float = Field(default=1.0, ge=0)
    optimizer_type: Literal["adamw", "adamw8bit"] = "adamw"
    adam_betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = Field(default=0.01, ge=0)
    scheduler_type: Literal["constant", "linear", "cosine", "cosine_with_restarts", "polynomial"] = "linear"
    scheduler_params: dict[str, Any] = Field(default_factory=dict)
    warmup_steps: int = Field(default=0, ge=0)
    enable_gradient_checkpointing: bool = True
    max_seq_tokens: int = Field(
        default=70_000,
        ge=1024,
        description="Pre-flight gate: samples whose packed sequence exceeds this are skipped before the "
        "forward pass (never between backward and step -- that breaks ZeRO-3's accumulation contract). "
        "~70k is the measured ceiling for LoRA on 80GB cards; 48GB cards want far less.",
    )


class AccelerationConfig(ConfigBaseModel):
    strategy: Literal["ddp", "model_parallel", "deepspeed_zero2", "deepspeed_zero3"] = Field(
        default="ddp",
        description=(
            "ddp: one full replica per GPU. Only viable with quantization on 48GB cards, but there is "
            "no cross-GPU parameter traffic at all.\n"
            "model_parallel: one process, bf16 weights split by transformer block across the GPUs "
            "(~8GB/GPU on 8 cards) with the index-consuming heads pinned. Full precision on small "
            "cards, at the cost of no data parallelism.\n"
            "deepspeed_zero3: shards weights, gradients and optimizer state across ranks. Each rank "
            "must be able to hold the model before partitioning, so it needs cards that fit 66GB "
            "(80GB class) unless you add deepspeed.zero.Init."
        ),
    )
    deepspeed_config: Path | None = Field(
        default=None,
        description="Explicit DeepSpeed JSON. When omitted a config is generated from this section.",
    )
    mixed_precision_mode: Literal["no", "fp16", "bf16"] = "bf16"
    quantization: Literal["none", "int8-quanto", "fp8-quanto", "nf4-bnb", "int8-bnb"] = Field(
        default="none",
        description="Quantize the frozen base weights (LoRA stays in bf16). nf4-bnb takes the 33B "
        "transformer to ~17GB, which is what makes single-GPU-replica DDP possible on a 48GB card.",
    )
    offload_optimizer_during_validation: bool = False

    @model_validator(mode="after")
    def _reject_quantized_zero3(self) -> AccelerationConfig:
        if self.quantization != "none" and self.strategy.startswith("deepspeed"):
            raise ValueError(
                "Quantized base weights and DeepSpeed ZeRO parameter sharding do not compose: ZeRO "
                "partitions and all-gathers raw parameter tensors, which quantized modules no longer "
                "expose. Use strategy: ddp with quantization, or drop quantization for ZeRO."
            )
        return self


class DataConfig(ConfigBaseModel):
    preprocessed_data_root: Path = Field(
        ..., description="The .precomputed directory written by scripts/process_dataset.py."
    )
    num_dataloader_workers: int = Field(default=2, ge=0)
    val_split_every: int = Field(
        default=20,
        ge=0,
        description="Deterministic held-out split: md5(sample_id) %% N == 0 goes to validation "
        "(20 -> ~5%%). 0 disables the split entirely.",
    )
    shuffle: bool = True


# =============================================================================
# Validation / checkpoints / logging
# =============================================================================


class ValidationConditionConfig(ConfigBaseModel):
    """A conditioning input for one validation sample, given as raw media."""

    type: Literal["first_frame", "last_frame", "reference"]
    image: str | None = None
    video: str | None = None
    audio: str | None = None

    @model_validator(mode="after")
    def _exactly_one_medium(self) -> ValidationConditionConfig:
        provided = [x for x in (self.image, self.video, self.audio) if x is not None]
        if len(provided) != 1:
            raise ValueError("Exactly one of image / video / audio must be set on a validation condition")
        if self.type in ("first_frame", "last_frame") and self.image is None:
            raise ValueError(f"'{self.type}' validation conditioning takes an image")
        return self


class ValidationSample(ConfigBaseModel):
    prompt: str
    conditions: list[ValidationConditionConfig] = Field(default_factory=list)
    video_dims: tuple[int, int, int] | None = Field(
        default=None, description="Per-sample (width, height, frames) override."
    )
    seed: int | None = None


class ValidationConfig(ConfigBaseModel):
    """Validation settings.

    Note what is *absent*: no negative prompt and no CFG scales. H3's released
    checkpoints are guidance-distilled -- guidance is baked into the weights, so
    every denoising step is a single forward pass with no unconditional branch.
    Audio is likewise not optional: H3 generates video and audio jointly from one
    packed sequence.
    """

    samples: list[ValidationSample] = Field(default_factory=list)
    video_dims: tuple[int, int, int] = Field(default=(704, 704, 107), description="(width, height, frames)")
    frame_rate: float = 24.0
    seed: int = 42
    inference_steps: int = Field(default=30, ge=1)
    interval: int | None = Field(default=250, description="Steps between validations; null disables.")
    skip_initial_validation: bool = False
    sample_media: bool = Field(
        default=True,
        description="Run real generation. Turning this off keeps the cheap held-out loss and skips the "
        "expensive denoising sweep -- useful when the pipeline does not fit alongside the optimizer.",
    )
    loss_sigmas: tuple[float, ...] = Field(
        default=(0.3, 0.6, 0.9),
        description="Fixed u grid for the held-out loss. With seeded noise this makes the validation "
        "curve a deterministic function of the weights, comparable across runs.",
    )
    max_loss_samples: int = Field(default=8, ge=0)

    @field_validator("video_dims")
    @classmethod
    def _check_geometry(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        width, height, frames = value
        # require_generatable, not just create: H3 refuses to generate outside
        # 5-15s, and finding that out mid-run wastes the whole validation pass.
        Geometry.create(width=width, height=height, num_frames=frames).require_generatable()
        return value


class CheckpointsConfig(ConfigBaseModel):
    interval: int | None = Field(default=250)
    keep_last_n: int = Field(default=-1, description="-1 keeps everything.")
    precision: Literal["bfloat16", "float32"] = "bfloat16"
    save_training_state: Literal["full", "minimal", "off"] = Field(
        default="full",
        description="full: optimizer + scheduler + step, so a resume is exact. minimal: weights + step. "
        "off: weights only. Under ZeRO-3 the gathered checkpoint is always weight-only.",
    )


class FlowMatchingConfig(ConfigBaseModel):
    timestep_sampling_mode: Literal["uniform", "logit_normal", "shifted_logit_normal"] = "uniform"
    timestep_sampling_params: dict[str, Any] = Field(default_factory=dict)
    video_shift: float = Field(
        default=DEFAULT_VIDEO_SHIFT,
        gt=0,
        description="Shift of the video noise schedule. H3 ships 12.0; changing it desynchronizes "
        "training from the inference scheduler.",
    )
    audio_shift: float = Field(default=DEFAULT_AUDIO_SHIFT, gt=0, description="Audio schedule shift (H3 ships 3.0).")


class HubConfig(ConfigBaseModel):
    push_to_hub: bool = False
    hub_model_id: str | None = None

    @model_validator(mode="after")
    def _need_id(self) -> HubConfig:
        if self.push_to_hub and not self.hub_model_id:
            raise ValueError("hub.hub_model_id is required when hub.push_to_hub is true")
        return self


class WandbConfig(ConfigBaseModel):
    enabled: bool = False
    project: str = "h3-trainer"
    entity: str | None = None
    name: str | None = None
    tags: list[str] = Field(default_factory=list)
    log_validation_videos: bool = True


class H3TrainerConfig(ConfigBaseModel):
    model: ModelConfig
    lora: LoraConfig = Field(default_factory=LoraConfig)
    training_strategy: FlexibleStrategyConfig = Field(default_factory=FlexibleStrategyConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    acceleration: AccelerationConfig = Field(default_factory=AccelerationConfig)
    data: DataConfig
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    checkpoints: CheckpointsConfig = Field(default_factory=CheckpointsConfig)
    flow_matching: FlowMatchingConfig = Field(default_factory=FlowMatchingConfig)
    hub: HubConfig = Field(default_factory=HubConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    seed: int = 42
    output_dir: Path = Path("outputs/h3_lora")

    @model_validator(mode="after")
    def _cross_section_checks(self) -> H3TrainerConfig:
        if self.training_strategy.uses_references and self.model.variant != "ref2va":
            raise ValueError(
                "Reference (IC-LoRA) conditioning requires model.variant: ref2va -- the FL2VA "
                "transformer has no reference rows in its packed layout."
            )
        if self.model.training_mode == "full" and self.acceleration.strategy != "deepspeed_zero3":
            raise ValueError(
                "training_mode: full means 33B trainable parameters plus optimizer state; that needs "
                "acceleration.strategy: deepspeed_zero3."
            )
        if self.optimization.enable_gradient_checkpointing and self.lora.dropout > 0:
            raise ValueError(
                "lora.dropout > 0 with gradient checkpointing gives a recomputed forward that differs "
                "from the original one. Set lora.dropout: 0.0."
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> H3TrainerConfig:
        with Path(path).open() as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ValueError(f"{path} does not contain a YAML mapping")
        return cls.model_validate(raw)

    def to_yaml(self, path: str | Path) -> None:
        payload = self.model_dump(mode="json")
        with Path(path).open("w") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, width=100)
