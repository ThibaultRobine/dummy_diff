# diffusion_model_manager_mine.py
import torch
from pathlib import Path
import datetime as dt
import dataclasses
from typing import Optional, Union

_model = None
_last_ckpt_path = None

CKPT_DIR = Path("./trained_diffusion_ckpts")
CKPT_DIR.mkdir(exist_ok=True, parents=True)

def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def _default_ckpt_path() -> Path:
    return CKPT_DIR / f"diffusion_{_timestamp()}.pt"

def set_trained_diffusion_model(model, ckpt_path: Optional[Union[str, Path]] = None) -> Path:
    global _model, _last_ckpt_path
    
    # 1. Handle checkpoint path
    if ckpt_path is None:
        ckpt_path = _default_ckpt_path()
    ckpt_path = Path(ckpt_path)
    
    # 2. Extract metadata from model
    torch.save({
            'model_state_dict': model.state_dict(),
            }, 'model_checkpoint_latent_linear_100_5000.pth')
    model.cuda()  # Move back to GPU if needed
    
    # 4. Update in-memory references
    _model = model
    _last_ckpt_path = ckpt_path
    
    return ckpt_path

def get_trained_diffusion_model():
    return _model

def last_checkpoint_path() -> Optional[Path]:
    return _last_ckpt_path
