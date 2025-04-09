# src/custom_wrapper.py

from improved_diffusion.unet import UNetModel
from improved_diffusion.script_util import create_gaussian_diffusion
from typing import Tuple

def parse_channel_mult(channel_mult_str: str) -> Tuple[int, ...]:
    """
    Convert a comma-separated string like '1,2,4,8'
    into a tuple of ints (1,2,4,8).
    """
    return tuple(int(x.strip()) for x in channel_mult_str.split(',') if x.strip())

def parse_attn_resolutions(attn_str: str, image_size: int):
    """
    For 1D signals, we treat 'attention_resolutions' as a comma-separated list
    of integer downsample factors. If empty, no attention layers.
    Example: "4,2" => attention at DS=4 and DS=2.
    """
    if not attn_str:
        return ()
    factors = parse_channel_mult(attn_str)  # e.g. (4,2)
    return factors

def create_custom_model_and_diffusion(
    *,
    # UNetModel hyperparams
    in_channels: int,
    out_channels: int,
    dims: int,
    image_size: int,          # For 1D signals, length dimension
    model_channels: int,
    channel_mult: str,
    num_res_blocks: int,
    attention_resolutions: str,
    dropout: float,
    class_cond: bool,

    # Diffusion hyperparams
    diffusion_steps: int,
    noise_schedule: str,
    learn_sigma: bool,
    sigma_small: bool,
    predict_xstart: bool,
    rescale_timesteps: bool,
    rescale_learned_sigmas: bool,
    use_kl: bool,
    timestep_respacing: str,

    # Additional optional
    num_heads: int = 1,
    num_heads_upsample: int = -1,
    use_checkpoint: bool = False,
    use_scale_shift_norm: bool = True,
):
    """
    A minimal wrapper that:
    1) Creates a UNetModel in 1D if dims=1.
    2) Creates a GaussianDiffusion object from improved_diffusion code.

    :param image_size: the length of your 1D signal. e.g. 16 => after 4 downsamples we get length=1.
    """
    # 1) Parse channel multiplier
    channel_mult_tuple = parse_channel_mult(channel_mult)

    # 2) Parse attention factors
    attn_factors = parse_attn_resolutions(attention_resolutions, image_size)

    # 3) Create the UNet
    model = UNetModel(
        in_channels=in_channels,
        model_channels=model_channels,
        out_channels=(out_channels if not learn_sigma else out_channels * 2),
        num_res_blocks=num_res_blocks,
        attention_resolutions=attn_factors,
        dropout=dropout,
        channel_mult=channel_mult_tuple,
        num_classes=None if not class_cond else 1000,  # official code uses 1000 for class_cond
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dims=dims,     # 1 => 1D conv in improved_diffusion
    )

    # 4) Create the diffusion object
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=(timestep_respacing or str(diffusion_steps)),
    )

    return model, diffusion
