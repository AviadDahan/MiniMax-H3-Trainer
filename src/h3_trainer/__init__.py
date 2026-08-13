"""MiniMax H3 Trainer — LoRA / IC-LoRA training for MiniMax-H3.

MiniMax-H3 is an omni-modal rectified-flow transformer that generates video and
synchronized stereo audio in a single packed sequence. Everything in this package
is organized around that packed sequence:

    [ text rows | conditioning rows | target audio rows | target video rows ]

`h3_trainer.packing` builds it, `h3_trainer.flow_matching` noises the target rows
and computes the loss, and `h3_trainer.training_strategies.flexible` decides which
rows are targets and which are conditioning.
"""

import logging
import os

__version__ = "0.1.0"

# The CUDA allocator config has to be in place before torch is first imported;
# long packed sequences fragment the caching allocator badly enough to OOM with
# 14+ GB nominally free. scripts/env.sh sets it for the shell, this is the
# belt-and-braces path for `python -c "import h3_trainer"` style entry.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _make_logger() -> logging.Logger:
    log = logging.getLogger("h3_trainer")
    if not log.handlers:
        try:
            from rich.logging import RichHandler

            handler: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
            handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        except ImportError:  # rich is optional at runtime
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(handler)
        log.setLevel(os.environ.get("H3_LOG_LEVEL", "INFO").upper())
        log.propagate = False
    return log


logger = _make_logger()

__all__ = ["logger", "__version__"]
