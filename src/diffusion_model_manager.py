# diffusion_model_manager.py
#
# Keep a pointer in memory **and** dump a checkpoint to disk
# every time we register a trained diffusion model.

from pathlib import Path
import torch
import datetime as _dt
from typing import Optional, Union

# ------------------------------------------------------------------
# In‑memory cache
_trained_model = None          # (model, diffusion) tuple
_last_ckpt_path: Optional[Path] = None
# ------------------------------------------------------------------

# Directory where checkpoints will be written
CKPT_DIR = Path("./trained_diffusion_ckpts")
CKPT_DIR.mkdir(exist_ok=True)

def _timestamp() -> str:
    """2025‑04‑17T15‑31‑08 → 20250417_153108"""
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def _default_ckpt_path() -> Path:
    return CKPT_DIR / f"diffusion_{_timestamp()}.pt"

# ------------------------------------------------------------------
def set_trained_diffusion_model(model, diffusion, ckpt_path: Optional[Union[str, Path]] = None):
    """
    Stores the pair (model, diffusion) in RAM **and** serialises the model
    weights to disk.  Returns the checkpoint path.

    Parameters
    ----------
    model : torch.nn.Module
        The trained UNet / diffusion model.
    diffusion : Any
        The diffusion object (kept in‑memory, usually rebuilt from kwargs).
    ckpt_path : str | Path | None
        Where to save.  If None, a timestamped file is created in CKPT_DIR.
    """
    global _trained_model, _last_ckpt_path

    if ckpt_path is None:
        ckpt_path = _default_ckpt_path()
    ckpt_path = Path(ckpt_path)

    # ------------------------------------------------------------------
    # 1.  Save weights (state_dict) – much smaller & portable
    #     You can save the whole object, but state_dict is recommended.
    # ------------------------------------------------------------------
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),    # to CPU → smaller file
            "meta": {
                "created": _timestamp(),
                "model_channels": getattr(model, "out_channels", None),
                # add anything else you want to remember
            },
        },
        ckpt_path,
    )
    # If you still need the model on GPU afterwards
    model.cuda()

    # ------------------------------------------------------------------
    # 2.  Cache in RAM so other parts of the pipeline can reuse it
    # ------------------------------------------------------------------
    _trained_model   = (model, diffusion)
    _last_ckpt_path  = ckpt_path
    return ckpt_path

def get_trained_diffusion_model():
    """
    Returns the cached (model, diffusion) tuple, or None if nothing cached.
    """
    return _trained_model

def last_checkpoint_path() -> Optional[Path]:
    """Handy helper if you need the filename later."""
    return _last_ckpt_path
