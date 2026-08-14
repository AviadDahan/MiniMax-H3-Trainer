"""The training loop.

Deliberately built on plain ``torch.distributed`` / DeepSpeed rather than a
higher-level wrapper: ZeRO-3 plus LoRA plus 10k-token packed sequences has enough
sharp edges (see the FIX comments below) that the ordering of operations needs to
be explicit and readable.

The ordering rules that matter:

* the sequence-length gate runs **before** the forward pass and is all-reduced so
  every rank skips the same step -- a rank that skips alone deadlocks the others
  on the next collective;
* ``backward`` and ``step`` stay strictly adjacent -- anything in between breaks
  ZeRO-3's gradient-accumulation bookkeeping;
* checkpoints are written **before** validation, because a validation forward
  leaves ZeRO-3 prefetch in flight and a parameter gather right after it fails
  with "Cannot partition a param in flight".
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from h3_trainer import logger
from h3_trainer.checkpointing import (
    TrainingState,
    apply_checkpoint,
    find_latest_checkpoint,
    load_checkpoint_weights,
    prune_checkpoints,
    save_checkpoint,
)
from h3_trainer.config import H3TrainerConfig
from h3_trainer.datasets import BucketBatchSampler, PrecomputedDataset, bucket_collate
from h3_trainer.flow_matching import SigmaPair
from h3_trainer.logging_utils import RunLogger
from h3_trainer.lora import set_trainable, trainable_parameter_count, trainable_state_dict
from h3_trainer.model_loader import enable_gradient_checkpointing, load_transformer
from h3_trainer.timestep_samplers import build_timestep_sampler
from h3_trainer.training_strategies import get_training_strategy


class DistributedContext:
    """Rank/device bookkeeping for single-GPU, DDP and DeepSpeed alike."""

    def __init__(self, strategy: str) -> None:
        self.strategy = strategy
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.distributed = self.world_size > 1
        self.device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        if self.distributed:
            if strategy.startswith("deepspeed"):
                import deepspeed

                deepspeed.init_distributed("nccl")
            elif not torch.distributed.is_initialized():
                torch.distributed.init_process_group("nccl")

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def all_reduce_max(self, value: float) -> float:
        if not self.distributed:
            return value
        tensor = torch.tensor([value], device=self.device)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
        return float(tensor.item())

    def all_reduce_min(self, value: int) -> int:
        if not self.distributed:
            return value
        tensor = torch.tensor([value], device=self.device)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
        return int(tensor.item())

    def all_reduce_mean(self, values: torch.Tensor) -> torch.Tensor:
        if not self.distributed:
            return values
        torch.distributed.all_reduce(values)
        return values / self.world_size

    def barrier(self) -> None:
        if self.distributed:
            torch.distributed.barrier()

    def shutdown(self) -> None:
        if self.distributed and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def deepspeed_config(config: H3TrainerConfig) -> dict[str, Any]:
    """Generate a DeepSpeed config from the trainer config.

    ``stage3_gather_16bit_weights_on_model_save`` stays off: we gather only the
    trainable tensors ourselves, and letting DeepSpeed materialize all 66GB on
    rank 0 is exactly the stall that trips the NCCL watchdog.
    """
    stage = 3 if config.acceleration.strategy == "deepspeed_zero3" else 2
    return {
        "bf16": {"enabled": config.acceleration.mixed_precision_mode == "bf16"},
        "fp16": {"enabled": config.acceleration.mixed_precision_mode == "fp16"},
        "train_micro_batch_size_per_gpu": config.optimization.batch_size,
        "gradient_accumulation_steps": config.optimization.gradient_accumulation_steps,
        "gradient_clipping": config.optimization.max_grad_norm,
        "zero_optimization": {
            "stage": stage,
            "overlap_comm": False,
            "contiguous_gradients": True,
            "reduce_bucket_size": 50_000_000,
            "stage3_prefetch_bucket_size": 50_000_000,
            "stage3_param_persistence_threshold": 100_000,
            "stage3_gather_16bit_weights_on_model_save": False,
        },
        "steps_per_print": 10**9,
        "wall_clock_breakdown": False,
    }


class H3Trainer:
    """Trains one adapter. ``config.output_dir`` is the *root*; each launch gets
    its own timestamped directory beneath it.

    Runs are hours long and get relaunched -- after a crash, a config change, or a
    corrected estimate. Writing every launch into one directory means the second
    one overwrites the first's ``train.log`` and interleaves its ``metrics.jsonl``,
    and checkpoints from different configurations end up side by side with nothing
    to tell them apart. Keeping each launch separate makes runs comparable and
    makes it safe to relaunch without hand-moving the previous attempt out of the
    way first.
    """

    def _new_run_dir(self, root: Path) -> Path:
        """``<root>/<UTC timestamp>``, identical on every rank.

        Ranks cannot each call ``time.time()``: they start milliseconds apart and
        would land in different directories, so rank 1 would write checkpoints
        where rank 0 never looks. Reducing to the minimum makes every rank adopt
        the earliest clock, which is a value they can all agree on without adding
        a broadcast.
        """
        stamp = self.context.all_reduce_min(int(time.time()))
        return root / datetime.fromtimestamp(stamp, tz=timezone.utc).strftime("%Y%m%d-%H%M%S")

    def _point_latest_at(self, run_dir: Path) -> None:
        """A stable path to the newest run, for tooling and symlinks elsewhere.

        Without it every consumer -- plot_metrics, the artifacts/ links, an
        evaluation script -- would need to know the timestamp of the run it wants.
        """
        latest = run_dir.parent / "latest"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(run_dir.name)
        except OSError as exc:  # a filesystem without symlinks should not end a run
            logger.warning("could not update %s: %s", latest, exc)

    def __init__(self, config: H3TrainerConfig) -> None:
        self.config = config
        self.context = DistributedContext(config.acceleration.strategy)
        self.state = TrainingState()
        self.run_root = Path(config.output_dir)
        self.output_dir = self._new_run_dir(self.run_root)
        if self.context.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._point_latest_at(self.output_dir)
        self.context.barrier()

        torch.manual_seed(config.seed + self.context.rank)
        self.strategy = get_training_strategy(config.training_strategy)
        self.timestep_sampler = build_timestep_sampler(
            config.flow_matching.timestep_sampling_mode,
            config.flow_matching.timestep_sampling_params,
        )
        self.run_logger = RunLogger(
            self.output_dir,
            wandb_config=config.wandb,
            config_snapshot=json.loads(config.model_dump_json()),
            is_main_process=self.context.is_main,
        )

        self.model = None
        self.engine = None
        self.optimizer = None
        self.scheduler = None
        self._skipped_long = 0
        # Console cadence: dense enough to watch, sparse enough to read.
        self.log_every = max(1, min(10, config.optimization.steps // 20 or 1))

    # ------------------------------------------------------------------ setup

    def setup(self) -> None:
        config = self.config
        # Validation prompts are encoded before the transformer is placed, while
        # the 63GB conditioner still has the machine to itself. Afterwards there
        # is no room for it beside the model on small cards.
        self._prepare_validation_media()
        use_deepspeed = config.acceleration.strategy.startswith("deepspeed")
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[
            config.acceleration.mixed_precision_mode
        ]

        model_parallel = config.acceleration.strategy == "model_parallel"
        if model_parallel:
            if self.context.distributed:
                raise RuntimeError(
                    "strategy: model_parallel runs a single process that owns every GPU -- launch it "
                    "with plain `python scripts/train.py`, not torchrun/deepspeed."
                )
            from h3_trainer.model_loader import load_sharded_transformer

            # One copy of the bf16 weights, split by transformer block across the
            # visible GPUs. ~8GB/GPU on 8 cards, versus 66GB for a single replica.
            self.model = load_sharded_transformer(
                config.model.model_path,
                variant=config.model.variant,
                dtype=dtype,
                primary=self.context.device.index or 0,
            )
        else:
            self.model = load_transformer(
                config.model.model_path,
                variant=config.model.variant,
                dtype=dtype,
                device=None if use_deepspeed else self.context.device,
                quantization=config.acceleration.quantization,
            )
            if not use_deepspeed and config.acceleration.quantization == "none":
                self.model = self.model.to(self.context.device)

        if config.optimization.enable_gradient_checkpointing:
            enable_gradient_checkpointing(self.model, use_deepspeed)

        set_trainable(self.model, config.model.training_mode, config.lora)

        # [FIX9] Resume BEFORE the model is wrapped or sharded: once ZeRO-3
        # partitions the parameters, load_state_dict sees shape-[0] shards and
        # loads nothing at all, silently.
        if config.model.load_checkpoint is not None:
            checkpoint = find_latest_checkpoint(config.model.load_checkpoint)
            if checkpoint is None:
                raise FileNotFoundError(f"No checkpoint found at {config.model.load_checkpoint}")
            weights, state = load_checkpoint_weights(checkpoint)
            apply_checkpoint(self.model, weights)
            self.state = state
            self.run_logger.message(f"Resumed from {checkpoint} at step {state.step}")

        if use_deepspeed:
            self.model = self.model.to(self.context.device)

        trainable, total = trainable_parameter_count(self.model)
        self.run_logger.message(
            f"{config.model.training_mode} training: {trainable:,} trainable of {total:,} parameters "
            f"({100 * trainable / max(total, 1):.3f}%), variant={config.model.variant}, "
            f"quantization={config.acceleration.quantization}, strategy={config.acceleration.strategy}"
        )
        if trainable == 0:
            raise RuntimeError("No trainable parameters -- check lora.target_modules / training_mode")

        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = self._build_optimizer(parameters)
        self.scheduler = self._build_scheduler(self.optimizer)

        if model_parallel:
            pass  # a single process already owns every device; nothing to wrap
        elif config.acceleration.strategy == "ddp" and self.context.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.context.local_rank],
                broadcast_buffers=False,
                # Conditioning probabilities below 1.0 mean some parameters go
                # unused on some steps; without this DDP hangs waiting for their
                # gradients.
                find_unused_parameters=True,
            )
        elif use_deepspeed:
            import deepspeed

            ds_config = deepspeed_config(config)
            if config.acceleration.deepspeed_config is not None:
                with Path(config.acceleration.deepspeed_config).open() as handle:
                    ds_config = json.load(handle)
            self.model, self.optimizer, _, _ = deepspeed.initialize(
                model=self.model,
                model_parameters=parameters,
                optimizer=self.optimizer,
                config=ds_config,
            )
            self.engine = self.model

        self._setup_data()

    def _build_optimizer(self, parameters: list[torch.nn.Parameter]) -> torch.optim.Optimizer:
        optimization = self.config.optimization
        if optimization.optimizer_type == "adamw8bit":
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(
                parameters,
                lr=optimization.learning_rate,
                betas=tuple(optimization.adam_betas),
                weight_decay=optimization.weight_decay,
            )
        return torch.optim.AdamW(
            parameters,
            lr=optimization.learning_rate,
            betas=tuple(optimization.adam_betas),
            weight_decay=optimization.weight_decay,
        )

    def _build_scheduler(self, optimizer: torch.optim.Optimizer):
        from torch.optim.lr_scheduler import LambdaLR

        optimization = self.config.optimization
        total = optimization.steps
        warmup = optimization.warmup_steps

        def factor(step: int) -> float:
            if warmup and step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            progress = min(max(progress, 0.0), 1.0)
            kind = optimization.scheduler_type
            if kind == "constant":
                return 1.0
            if kind == "linear":
                return 1.0 - progress
            if kind == "cosine":
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            if kind == "cosine_with_restarts":
                cycles = float(optimization.scheduler_params.get("num_cycles", 1))
                return 0.5 * (1.0 + math.cos(math.pi * ((cycles * progress) % 1.0)))
            if kind == "polynomial":
                power = float(optimization.scheduler_params.get("power", 1.0))
                return (1.0 - progress) ** power
            return 1.0

        return LambdaLR(optimizer, factor)

    def _setup_data(self) -> None:
        config = self.config
        sources = config.training_strategy.data_sources()
        root = config.data.preprocessed_data_root

        self.train_dataset = PrecomputedDataset(
            root,
            sources,
            split="train",
            val_split_every=config.data.val_split_every,
        )
        self.val_dataset = None
        if config.data.val_split_every > 0:
            try:
                self.val_dataset = PrecomputedDataset(
                    root, sources, split="val", val_split_every=config.data.val_split_every
                )
            except RuntimeError:
                self.val_dataset = None

        sampler = BucketBatchSampler(
            self.train_dataset,
            batch_size=config.optimization.batch_size,
            shuffle=config.data.shuffle,
            seed=config.seed,
        )
        self.sampler = sampler
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=config.data.num_dataloader_workers,
            collate_fn=bucket_collate,
            pin_memory=True,
        )
        self.run_logger.message(
            f"Data: {len(self.train_dataset)} train / "
            f"{0 if self.val_dataset is None else len(self.val_dataset)} val samples, "
            f"{len(sampler.buckets)} layout bucket(s)"
        )

    # ------------------------------------------------------------------ train

    def train(self) -> None:
        config = self.config
        self.model.train()
        dtype = torch.bfloat16 if config.acceleration.mixed_precision_mode == "bf16" else torch.float32
        total_steps = config.optimization.steps
        accumulation = config.optimization.gradient_accumulation_steps
        micro_step = 0
        started = time.time()

        if self.val_dataset and not config.validation.skip_initial_validation:
            self.validate(self.state.step)

        while self.state.step < total_steps:
            self.sampler.set_epoch(self.state.epoch)
            executed_this_epoch = 0

            for batch in self.dataloader:
                if self.state.step >= total_steps:
                    break

                sigmas = SigmaPair.from_u(
                    self.timestep_sampler.sample(),
                    video_shift=config.flow_matching.video_shift,
                    audio_shift=config.flow_matching.audio_shift,
                )
                # [FIX7] Decide to skip *before* the forward pass, and agree
                # across ranks, so nobody is left waiting on a collective.
                record = batch["record"]
                estimated = record.sequence_length
                skip = self.context.all_reduce_max(
                    1.0 if estimated > config.optimization.max_seq_tokens else 0.0
                )
                if skip > 0:
                    self._skipped_long += 1
                    self._skipped_total += 1
                    continue

                packed = self.strategy.prepare(batch, sigmas, self.context.device, dtype)
                prediction_video, prediction_audio = self._forward(packed)
                loss = self.strategy.compute_loss(prediction_video, prediction_audio, packed)

                # backward and step stay strictly adjacent -- see module docstring.
                if self.engine is not None:
                    self.engine.backward(loss.total)
                    self.engine.step()
                    stepped = self.engine.is_gradient_accumulation_boundary()
                    grad_norm = float(self.engine.get_global_grad_norm() or 0.0)
                else:
                    (loss.total / accumulation).backward()
                    micro_step += 1
                    stepped = micro_step % accumulation == 0
                    grad_norm = 0.0
                    if stepped:
                        parameters = [p for p in self._core_model().parameters() if p.requires_grad]
                        grad_norm = float(
                            torch.nn.utils.clip_grad_norm_(parameters, config.optimization.max_grad_norm)
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)

                if not stepped:
                    continue

                self.scheduler.step()
                self.state.step += 1
                self.state.samples_seen += config.optimization.batch_size * self.context.world_size
                executed_this_epoch += 1

                if self.context.is_main:
                    metrics = loss.as_log_dict()
                    metrics.update(
                        {
                            "lr": float(self.scheduler.get_last_lr()[0]),
                            "grad_norm": grad_norm,
                            "sigma_video": sigmas.video,
                            "sigma_audio": sigmas.audio,
                            "u": sigmas.u,
                            "steps_per_sec": self.state.step / max(1e-6, time.time() - started),
                            "vram_gb": torch.cuda.max_memory_allocated() / 1e9
                            if torch.cuda.is_available()
                            else 0.0,
                            "skipped_long": self._skipped_long,
                        }
                    )
                    # Console every `log_every` steps; the file log keeps all of them.
                    self.run_logger.log(
                        metrics,
                        self.state.step,
                        prefix="train/",
                        console=self.state.step % self.log_every == 0 or self.state.step == 1,
                    )

                # Checkpoint before validation: a validation forward leaves ZeRO-3
                # prefetch in flight and the following gather fails.
                if self._should(config.checkpoints.interval):
                    self.save(self.state.step)
                if self.val_dataset and self._should(config.validation.interval):
                    self.validate(self.state.step)

            self.state.epoch += 1
            if executed_this_epoch == 0:
                raise RuntimeError(
                    f"An entire epoch executed zero steps -- every sample exceeded "
                    f"optimization.max_seq_tokens ({config.optimization.max_seq_tokens}). "
                    f"Lower the resolution bucket or raise the budget."
                )

        self.save(self.state.step)
        self.run_logger.summary(
            {
                "final_step": self.state.step,
                "samples_seen": self.state.samples_seen,
                "skipped_long": self._skipped_long,
                "wall_clock_hours": (time.time() - started) / 3600,
            }
        )
        self.run_logger.close()
        self.context.shutdown()

    def _should(self, interval: int | None) -> bool:
        return bool(interval) and self.state.step % int(interval) == 0

    def _core_model(self) -> torch.nn.Module:
        model = self.model
        if self.engine is not None:
            return self.engine.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            return model.module
        return model

    def _forward(self, packed) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(**packed.to_model_kwargs())

    # ------------------------------------------------------------- validation

    @torch.no_grad()
    def validate(self, step: int) -> None:
        """Held-out loss on a fixed sigma grid with seeded noise.

        Every rank must run the same number of forwards: under ZeRO-3 a forward
        is a collective, so a rank with fewer validation samples would desync the
        whole group. Hence the global minimum below.
        """
        if self.val_dataset is None or len(self.val_dataset) == 0:
            return
        config = self.config
        dtype = torch.bfloat16 if config.acceleration.mixed_precision_mode == "bf16" else torch.float32
        self.model.eval()

        local_count = min(len(self.val_dataset), config.validation.max_loss_samples)
        count = self.context.all_reduce_min(local_count)

        totals = torch.zeros(3, device=self.context.device)
        per_sigma: dict[float, torch.Tensor] = {
            u: torch.zeros(2, device=self.context.device) for u in config.validation.loss_sigmas
        }

        for index in range(count):
            sample = self.val_dataset[index]
            for u in config.validation.loss_sigmas:
                sigmas = SigmaPair.from_u(
                    u,
                    video_shift=config.flow_matching.video_shift,
                    audio_shift=config.flow_matching.audio_shift,
                )
                seed = abs(hash((sample["id"], u))) % (2**31)
                packed = self.strategy.prepare(
                    sample, sigmas, self.context.device, dtype, noise_seed=seed
                )
                prediction_video, prediction_audio = self._forward(packed)
                loss = self.strategy.compute_loss(prediction_video, prediction_audio, packed)
                totals += torch.stack(
                    [
                        loss.total.detach().float(),
                        loss.video.detach().float(),
                        torch.ones((), device=self.context.device),
                    ]
                )
                per_sigma[u] += torch.stack(
                    [loss.video.detach().float(), torch.ones((), device=self.context.device)]
                )

        if self.context.distributed:
            torch.distributed.all_reduce(totals)
            for tensor in per_sigma.values():
                torch.distributed.all_reduce(tensor)

        if self.context.is_main and float(totals[2]) > 0:
            metrics = {
                "loss": float(totals[0] / totals[2]),
                "loss_video": float(totals[1] / totals[2]),
            }
            for u, tensor in per_sigma.items():
                metrics[f"loss_video_u{u}"] = float(tensor[0] / tensor[1].clamp(min=1))
            self.run_logger.log(metrics, step, prefix="val/", console=True)

        if config.validation.sample_media and config.validation.samples:
            self._sample_validation_media(step)

        self.model.train()

    def _prepare_validation_media(self) -> None:
        """Encode the validation prompts once, before the model takes the GPUs."""
        config = self.config.validation
        if not (config.sample_media and config.samples) or not self.context.is_main:
            return
        try:
            from h3_trainer.validation_runner import ValidationRunner

            self._validation_runner = ValidationRunner(self.config, None, self.context.device)
            self._validation_runner.prepare()
        except Exception as exc:  # pragma: no cover - never block a run on this
            logger.warning("Validation media preparation failed (%s); sampling may be skipped", exc)
            self._validation_runner = None

    def _sample_validation_media(self, step: int) -> None:
        """Generate validation clips with the adapter applied."""
        if not self.context.is_main:
            return
        runner = getattr(self, "_validation_runner", None)
        if runner is None:
            try:
                from h3_trainer.validation_runner import ValidationRunner
            except ImportError as exc:  # pragma: no cover
                logger.warning("Validation sampling unavailable: %s", exc)
                return
            runner = ValidationRunner(self.config, self._core_model(), self.context.device)
            self._validation_runner = runner
        # The model is only available once setup() has run, so bind it here.
        runner.transformer = self._core_model()
        for index, path in enumerate(runner.run(step)):
            self.run_logger.log_media(
                f"val/sample_{index}", path, step, caption=self.config.validation.samples[index].prompt[:200]
            )

    # ------------------------------------------------------------ checkpoints

    def save(self, step: int) -> None:
        config = self.config
        dtype = torch.float32 if config.checkpoints.precision == "float32" else torch.bfloat16
        path = save_checkpoint(
            self.output_dir,
            step,
            self._core_model(),
            self.state,
            engine=self.engine,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            dtype=dtype,
            save_training_state=config.checkpoints.save_training_state,
            config_snapshot=json.loads(config.model_dump_json()),
            is_main_process=self.context.is_main,
        )
        self.context.barrier()
        if path is not None and self.context.is_main:
            prune_checkpoints(self.output_dir, config.checkpoints.keep_last_n)
            self._export_adapter(path)

    def _export_adapter(self, checkpoint_path: Path) -> None:
        """Also write a ComfyUI-loadable adapter next to the raw checkpoint."""
        if self.config.model.training_mode != "lora":
            return
        from h3_trainer.lora import export_lora

        try:
            state = trainable_state_dict(self._core_model())
            export_lora(
                state,
                checkpoint_path / "lora_comfyui.safetensors",
                metadata={
                    "h3_trainer_version": __import__("h3_trainer").__version__,
                    "variant": self.config.model.variant,
                    "rank": str(self.config.lora.rank),
                    "alpha": str(self.config.lora.alpha),
                },
            )
        except Exception as exc:
            logger.warning("ComfyUI export failed (the raw checkpoint is still valid): %s", exc)


def train_from_config(config_path: str | Path) -> None:
    config = H3TrainerConfig.from_yaml(config_path)
    trainer = H3Trainer(config)
    trainer.setup()
    trainer.train()
