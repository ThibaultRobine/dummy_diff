import torch
from torch import optim
import pytorch_warmup as warmup
import dataclasses
from typing import Union

# Project‑local imports ---------------------------------------------------------
from diffusion_model_manager_mine import (
    get_trained_diffusion_model,
    set_trained_diffusion_model,
)
from model_unet import (
    DiscreteDDPMProcess,
    UniformDiscreteTimeSampler,
    AdaptiveTimeSampler,   
    DiffusionModel,
)
@dataclasses.dataclass
class NetConfig:
    activation: str
    time_embedding_dim: int
# -----------------------------------------------------------------------------
# Public helper ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_or_load_diffusion_model(
    diffusion_args: dict,
    feature_bank: torch.Tensor,
    device: Union[str, torch.device] = "cuda",
) -> DiffusionModel:
    """Return a trained ``DiffusionModel`` instance.

    If another post‑processor has already trained one in this process, the
    cached copy from ``diffusion_model_manager_mine`` is returned and **no**
    extra work is done.

    Parameters
    ----------
    diffusion_args : dict
        Section of the YAML/JSON config that describes the diffusion model
        (schedule, UNet width, learning‑rate schedule, etc.).
    feature_bank : torch.Tensor  [N, D]
        All ID features (already normalised) available for training.
    device : str | torch.device, default "cuda"
        Where to place both the model and the training minibatches.

    Notes
    -----
    The routine reproduces exactly the block that every post‑processor had
    before, so replacing the in‑line code is a pure refactor.
    """

    # ------------------------------------------------------------------
    # 0. Return cached model if it exists
    # ------------------------------------------------------------------
    shared = get_trained_diffusion_model()
    if shared is not None:
        return shared

    # ------------------------------------------------------------------
    # 1. Parse high‑level config ------------------------------------------------
    # ------------------------------------------------------------------
    unet_cfg = diffusion_args.get("unet", {})
    net_cfg  = NetConfig(
        activation         = unet_cfg.get("activation", "relu"),
        time_embedding_dim = unet_cfg.get("time_embedding_dim", 128),
    )

    # forward process ---------------------------------------------------
    proc = DiscreteDDPMProcess(
        num_diffusion_timesteps = diffusion_args["num_diffusion_timesteps"],
        schedule_type           = diffusion_args["schedule_type"],
    )

    # choose time‑sampler ----------------------------------------------
    if diffusion_args["schedule_type"] == "adaptive":
        t_sampler = AdaptiveTimeSampler(proc)
    else:
        t_sampler = UniformDiscreteTimeSampler(proc.tmin, proc.tmax)

    # main model --------------------------------------------------------
    model = DiffusionModel(
        diffusion_process = proc,
        time_sampler      = t_sampler,
        net_config        = net_cfg,
        data_shape        = (feature_bank.shape[1],),
    ).to(device)

    # ------------------------------------------------------------------
    # 2. Optimiser & LR schedule ---------------------------------------
    # ------------------------------------------------------------------
    train_cfg   = diffusion_args["train_loop"]
    batch_size  = train_cfg["batch_size"]

    optimiser   = optim.Adam(model.parameters(), lr=train_cfg["lr"])
    lr_sched    = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimiser,
        T_0    = train_cfg["total_steps"],
        eta_min= train_cfg.get("eta_min", 1e-12),
    )
    warm_sched  = warmup.LinearWarmup(optimiser, train_cfg["warmup_steps"])

    # utility: random batch from bank ----------------------------------
    def _sample_batch(x: torch.Tensor, n: int) -> torch.Tensor:
        idx = torch.randint(0, x.size(0), (n,), device=x.device)
        return x[idx]

    # ------------------------------------------------------------------
    # 3. Training loop --------------------------------------------------
    # ------------------------------------------------------------------
    model.train()
    feature_bank = feature_bank.to(device)
    for step in range(train_cfg["total_steps"]):
        x0 = _sample_batch(feature_bank, batch_size)
        optimiser.zero_grad()
        loss = model.loss(x0)
        loss.backward()
        optimiser.step()

        # console log every 50 steps
        if step % 50 == 0:
            print(f"[diffusion] step {step:6d} | loss {loss.item():.5f}")

        # LR schedule / warm‑up ----------------------------------------
        with warm_sched.dampening():
            if warm_sched.last_step + 1 >= train_cfg["warmup_steps"]:
                lr_sched.step()
            if warm_sched.last_step + 1 >= train_cfg.get("max_step", 10**9):
                break

    model.eval()
    set_trained_diffusion_model(model)
    return model
