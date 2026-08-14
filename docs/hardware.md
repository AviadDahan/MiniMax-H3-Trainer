# Hardware and cost

H3 is 33B - 66GB in bf16 - and the conditioner is another 63GB. On 48GB cards the right answer isn't
the obvious one.

## Acceleration strategies

Measured on 8×A6000 (no NVLink):

| strategy | weights/GPU | 48GB? | notes |
|---|---|---|---|
| `model_parallel` | ~8GB (8 GPUs), ~22GB (3) | **yes** | one process, blocks split across GPUs, full bf16. Floor is 2 GPUs - 66GB cannot fit on one. **This is the path every run here used.** |
| `ddp` + `nf4-bnb` | ~18GB | yes | full replicas; smoke-tested only, and see the 4-bit warning below |
| `ddp` + `int8-quanto` ⚠️ | ~33GB (estimated) | untested | never run; little room for activations at real resolutions |
| `deepspeed_zero3` | ~8GB after partition | **no** | each rank holds all 66GB *before* partitioning. Fails here; untested on 80GB cards. |

Model-parallel runs blocks **sequentially**, so extra GPUs buy memory, not speed. The reason to use
fewer GPUs per run is *concurrency* - four 2-GPU runs instead of one 4-GPU run.

Strategy is orthogonal to training mode; see
[configuration-reference.md](configuration-reference.md#acceleration) for the keys.

## Sequence length is the cost driver, and it is quadratic

Attention has no mask over the packed sequence, so everything attends to everything. Measured on
4 GPUs:

| what | rows | VRAM/GPU | step |
|---|---|---|---|
| 512×512×124, no reference | 9,970 | 23 GB | ~19 s |
| 448×768×124 **+ a reference video** | 27,364 | 31 GB | ~82 s |

A reference costs as many rows as the target it conditions, so an IC-LoRA sequence is roughly twice a
plain one before resolution is even considered - and 2.7× the rows came out at 4.3× the time. Plan
control-adapter runs around a bucket you can afford, not the largest one that fits in memory.

Multiply rows × step time × step count *before* launching. `optimization.max_seq_tokens` gates
over-budget samples before the forward pass; check the `skipped_long` counter so you know how much of
the dataset is being dropped.

## 4-bit is for training, not for looking at

NF4 fits the model on one card, but quantizing H3's AdaLN branches destroys generation - an NF4
sample here decoded to noise indistinguishable from decoding random latents. Train against it if you
must; evaluate against bf16 (`--placement shard`).

## <a name="untested"></a>What is implemented but untested

Written, reachable from config, and **never run** - treat as experimental and read the code before
relying on it. Listed rather than removed because the code paths exist and are probably close; they
simply have never been run.

| area | untested |
|---|---|
| training | `training_mode: full` and `heads`; `optimizer_type: adamw8bit` |
| acceleration | `deepspeed_zero2`; `deepspeed_zero3` (cannot start on 48GB, unverified on 80GB); `int8-quanto`, `int8-bnb`, `fp8-quanto` |
| modes | `a2v` and first+last-frame - expressible, no shipped config |
| inference | `--placement bf16` and `--placement offload` (`shard` and `quantize` are exercised) |
| logging | W&B **online** (offline is what every run here used) |
| publishing | `hub.push_to_hub` |
| scale | anything past 36 clips, ~10k rows, or a single machine |

Everything not in this list has at least one real run behind it. `nf4-bnb` has run, and is documented
as unusable for generation.
