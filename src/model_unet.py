import torch
import math
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bcast_right(x: torch.Tensor, ndim: int) -> torch.Tensor:
    """Util function for broadcasting to the right."""
    if x.ndim > ndim:
        raise ValueError(f'Cannot broadcast a value with {x.ndim} dims to {ndim} dims.')
    elif x.ndim < ndim:
        difference = ndim - x.ndim
        return x.view(x.shape + (1,) * difference)
    else:
        return x
    
class DiscreteDDPMProcess:
    """A Gaussian diffusion process: q(xt|x0) = N(sqrt_alpha_bar(t)*x0, sigma(t)^2 * I),
    which implies the following transition from x0 to xt:

    xt = sqrt_alpha_bar(t) x0 + sigma(t) eps, eps ~ N(0, I).

    Diffusion processes differ in how they specify sqrt_alpha_bar(t) and/or sigma(t).
    Here we follow the DDPM paper.

    """
    def __init__(
        self,
        num_diffusion_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_type: str = 'adaptive',
        lambda_min: float = -10.0,   
        lambda_max: float =  20.0   
    ):
        self._num_diffusion_timesteps = num_diffusion_timesteps
        self._schedule_type = schedule_type

        if self._schedule_type in ('linear', 'adaptive'):
            self._beta_start = beta_start
            self._beta_end = beta_end
            self._betas = np.linspace(self._beta_start, self._beta_end, self._num_diffusion_timesteps)
        elif self._schedule_type == 'cosine':
            s = 0.008
            steps = self._num_diffusion_timesteps + 1
            x = np.linspace(0, self._num_diffusion_timesteps, steps)
            alpha_bar = np.cos((x / self._num_diffusion_timesteps + s) / (1 + s) * np.pi * 0.5) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            self._betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
        elif self._schedule_type == 'linear_snr':                       # NEW
            # λ = log SNR; equally-space it → balances gradient variance
            lambdas = np.linspace(
                lambda_max, lambda_min, self._num_diffusion_timesteps + 1
            )
            alpha_bar = 1.0 / (1.0 + np.exp(-lambdas))                # ᾱ = sigmoid(λ)
            self._betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])      # β_t
            print(f"Adaptive schedule: {self._betas}")
            self._alpha_bar_precomputed = alpha_bar                   # save for later
        else:
            raise ValueError(f"Unsupported schedule type: {self._schedule_type}")

        alphas_bar = self._get_alphas_bar()
        #put in dtype=torch.float32
        self._sqrt_alphas_bar = torch.sqrt(torch.tensor(alphas_bar, dtype=torch.float32,device=device))
        #put in dtype=torch.float32
        self._sigmas = torch.sqrt(torch.tensor((1.0 - alphas_bar), dtype=torch.float32,device=device))

    @property
    def tmin(self):
        return 1

    @property
    def tmax(self):
        return self._num_diffusion_timesteps

    def _get_alphas_bar(self) -> np.ndarray:
        if self._schedule_type == 'linear':
            alphas = 1.0 - self._betas
            alphas_bar = np.cumprod(alphas)
            return np.concatenate([[1.0], alphas_bar])  # Add initial 1.0 for t=0
        elif self._schedule_type == 'cosine':
            # Return pre-computed alpha_bar from cosine schedule
            s = 0.008
            steps = self._num_diffusion_timesteps + 1
            x = np.linspace(0, self._num_diffusion_timesteps, steps)
            alpha_bar = np.cos((x / self._num_diffusion_timesteps + s) / (1 + s) * np.pi * 0.5) ** 2
            return alpha_bar / alpha_bar[0]  # Return normalized alpha_bar
        elif self._schedule_type == 'adaptive':          # ← NEW
            # use the exact same formula as the linear ramp
            alphas = 1.0 - self._betas
            alphas_bar = np.cumprod(alphas)
            return np.concatenate([[1.0], alphas_bar])
    
    def sqrt_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        return self._sqrt_alphas_bar[t.long()]

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self._sigmas[t.long()]

    def sample(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Draws samples from the forward diffusion process q(xt|x0)."""
        return bcast_right(self.sqrt_alpha_bar(t), x0.ndim) * x0 + bcast_right(self.sigma(t), x0.ndim) * eps
    

import torch.nn as nn

class UNet1D(nn.Module):
    """U-Net architecture for 1D data."""
    def __init__(self, in_channels, out_channels, time_embedding_dim, activation='relu'):
        super(UNet1D, self).__init__()
        self.activation = activation

        # Define the activation function
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif self.activation == 'elu':
            self.act = nn.ELU()
        else:
            self.act = nn.ReLU()  # default

        # Encoder layers
        self.down1 = nn.Conv1d(in_channels, 64, kernel_size=4, stride=2, padding=1)  # -> (batch, 64, 1024)
        self.down2 = nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1)          # -> (batch, 128, 512)
        self.down3 = nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1)         # -> (batch, 256, 256)
        self.down4 = nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1)         # -> (batch, 512, 128)

        # Time embedding layers
        self.time_emb = nn.Linear(time_embedding_dim, 512)

        # Decoder layers
        self.up1 = nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1)  # -> (batch, 256, 256)
        self.up2 = nn.ConvTranspose1d(512, 128, kernel_size=4, stride=2, padding=1)  # -> (batch, 128, 512)
        self.up3 = nn.ConvTranspose1d(256, 64, kernel_size=4, stride=2, padding=1)   # -> (batch, 64, 1024)
        self.up4 = nn.ConvTranspose1d(128, out_channels, kernel_size=4, stride=2, padding=1)  # -> (batch, out_channels, 2048)

    def forward(self, x, time_emb):
        # Encoder
        x1 = self.act(self.down1(x))  # (batch, 64, 1024)
        x2 = self.act(self.down2(x1)) # (batch, 128, 512)
        x3 = self.act(self.down3(x2)) # (batch, 256, 256)
        x4 = self.act(self.down4(x3)) # (batch, 512, 128)

        # Add time embedding to bottleneck
        time_emb = self.time_emb(time_emb).unsqueeze(2)  # (batch, 512, 1)
        x4 = x4 + time_emb  # Broadcasting over the length dimension

        # Decoder
        x = self.act(self.up1(x4))         # (batch, 256, 256)
        x = torch.cat([x, x3], dim=1)      # (batch, 512, 256)
        x = self.act(self.up2(x))          # (batch, 128, 512)
        x = torch.cat([x, x2], dim=1)      # (batch, 256, 512)
        x = self.act(self.up3(x))          # (batch, 64, 1024)
        x = torch.cat([x, x1], dim=1)      # (batch, 128, 1024)
        x = self.up4(x)                    # (batch, out_channels, 2048)
        return x

# Modified Net class to use the U-Net
class Net(nn.Module):
    """Combines U-Net and time embeddings."""
    def __init__(self, net_config, name: str = None):
        super(Net, self).__init__()

        self._time_encoder = SinusoidalTimeEmbedding(net_config.time_embedding_dim)
        self._predictor = UNet1D(
            in_channels=1,
            out_channels=1,  # Assuming the output is noise of the same shape
            time_embedding_dim=net_config.time_embedding_dim,
            activation=net_config.activation
        )

    def forward(self, noisy_data: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # Reshape noisy_data to (batch_size, channels, length)
        x = noisy_data.unsqueeze(1)  # Assuming input is (batch_size, 2048)
        time_embedding = self._time_encoder(time)
        outputs = self._predictor(x, time_embedding)
        # Reshape outputs back to (batch_size, length)
        outputs = outputs.squeeze(1)
        return outputs
    
class SinusoidalTimeEmbedding(nn.Module):
    """Time (positional) embedding as in Transformers."""

    def __init__(self, num_features: int, name: str = None):
        super(SinusoidalTimeEmbedding, self).__init__()
        self._num_features = num_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert len(inputs.shape) == 1
        half_dim = self._num_features // 2
        e = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embedding = torch.exp(-e * torch.arange(half_dim).float()).to(device)
        embedding = inputs.view(-1, 1) * embedding
        embedding = torch.cat([torch.cos(embedding), torch.sin(embedding)], dim=-1)
        if self._num_features % 2 == 1:
            embedding = nn.functional.pad(embedding, (0, 1))
        return embedding


from typing import Sequence

class UniformDiscreteTimeSampler:

    def __init__(self, tmin: int, tmax: int):
        self._tmin = tmin
        self._tmax = tmax

    def sample(self, shape: Sequence[int]) -> torch.Tensor:
        return torch.randint(low=self._tmin, high=self._tmax, size=shape)

class AdaptiveTimeSampler:
    """EMA schedule of Kingma & Gao, Appendix F."""
    def __init__(self, process, nbins=100, decay=0.999):
        self._T   = process.tmax
        self.decay = decay
        self.nbins = nbins
        # pre-compute log-SNR for every t ∈ [1,T]
        lam = torch.log(process._sqrt_alphas_bar[1:]**2 /
                        process._sigmas[1:]**2)            # shape [T]
        self._λ_lut = lam                                   # cache
        self.edges  = torch.linspace(lam.min(), lam.max(),
                                     nbins + 1, device=device)
        self.ema    = torch.ones(nbins, device=device)
        self._refresh_cdf()

    def _refresh_cdf(self):
        pdf = self.ema / self.ema.sum()
        self.cdf = torch.cumsum(pdf, 0)
        self.cdf[-1] = 1.0

    def sample(self, shape):
        u = torch.rand(shape, device=device)
        b = torch.searchsorted(self.cdf, u)                 # bin index
        b = torch.clamp(b, max=self.nbins-1)      # ← add this
        λ = self.edges[b] + (self.edges[b+1]-self.edges[b]) * torch.rand_like(u)
        t = (λ.unsqueeze(-1) - self._λ_lut).abs().argmin(-1) + 1
        return t.long()

    @torch.no_grad()
    def update(self, λ, per_item_se):
        idx = torch.bucketize(λ, self.edges) - 1
        for b in range(self.ema.numel()):
            m = (idx == b)
            if m.any():
                self.ema[b] = self.decay * self.ema[b] + (1-self.decay)*per_item_se[m].mean()
        self._refresh_cdf()
    
from typing import Tuple
class DiffusionModel(nn.Module):
    """Diffusion model."""

    def __init__(self, diffusion_process, time_sampler, net_config, data_shape):
        super(DiffusionModel, self).__init__()

        self._process = diffusion_process
        self._time_sampler = time_sampler
        self._net_config = net_config
        self._data_shape = data_shape
        self.net_fwd = Net(net_config)

    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        """Computes MSE between the true noise and predicted noise,
        i.e. the goal of the network is to correctly predict eps from a noisy observation
        xt = alpha(t) * x0 + sigma(t)**2 * eps"""

        t = self._time_sampler.sample((x0.shape[0],)).to(device)

        eps = torch.randn_like(x0).to(device)

        xt = self._process.sample(x0, t, eps)

        net_outputs = self.net_fwd(xt, t)

        per_item  = (net_outputs - eps).pow(2).mean(1)
        if hasattr(self._time_sampler, 'update'):              # NEW
            lam = torch.log(self._process.sqrt_alpha_bar(t)**2 /
                            self._process.sigma(t)**2)
            self._time_sampler.update(lam.detach(), per_item.detach())


        loss = per_item.mean()

        return loss

    #Used for sampling
    def _reverse_process_step(
        self,
        xt: torch.Tensor,
        t: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes parameters of a Gaussian p_{\theta}(x_{t-1}| x_t)."""

        t = t * torch.ones((xt.shape[0],), dtype=torch.int32, device=xt.device)

        # predict epsilon from x_t
        eps_pred = self.net_fwd(xt, t)
        # use self._sqrt_alpha_bar
        sqrt_alpha_t = self._process._sqrt_alphas_bar[t]/self._process._sqrt_alphas_bar[t-1]
        inv_sqrt_alpha_t = bcast_right(1/sqrt_alpha_t, xt.ndim)

        beta_t = torch.tensor(self._process._betas[t.cpu()-1]).to(device=xt.device,dtype=torch.float32)
        beta_t = bcast_right(beta_t, xt.ndim)

        inv_sigma_t = 1 / self._process.sigma(t)
        inv_sigma_t = bcast_right(inv_sigma_t, xt.ndim)


        mean = inv_sqrt_alpha_t * (xt - beta_t * inv_sigma_t* eps_pred)

        # DDPM instructs to use either the variance of the forward process
        # or the variance of q(x_{t-1}|x_t, x_0). Former is easier.
        std = torch.sqrt(beta_t)

        eps = torch.randn_like(xt)

        return mean, std, eps


    def sample(self, x0, sample_size):
        """To generate samples from DDPM, we follow the reverse process.
        At each step of the chain, we sample x_{t-1} from p(x_{t-1}| x_t, x0_pred) until we get to x_0."""
        with torch.no_grad():
            x = torch.randn((sample_size,) + self._data_shape, device=x0.device) #sample pure noise

            for t in range(self._process.tmax, 0, -1):
                mean, std, eps = self._reverse_process_step(x, t)
                if t == 1:
                    x = mean
                else:
                    x = mean + std * eps

        return x
    def plms_sample(self, x0, sample_size, steps=100):
        """PLMS sampling with numerical stability checks"""
        with torch.no_grad():
            # Initialize with random noise
            x = torch.randn((sample_size,) + self._data_shape, device=x0.device)
            
            # Create time steps
            timesteps = np.linspace(self._process.tmax, 1, steps, dtype=int)
            eps_history = []
            
            # Numerical stability parameters
            eps = 1e-8  # Small epsilon to prevent division by zero
            clamp_threshold = 1e3  # Threshold for gradient clipping

            for i, t in enumerate(timesteps):
                t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
                
                # 1. Predict noise using current x and timestep
                eps_pred = self.net_fwd(x, t_tensor)
                
                # Store first 3 predictions for PLMS initialization
                if i < 3:
                    eps_history.append(eps_pred)
                    continue
                    
                # Maintain history of last 4 predictions
                eps_history.append(eps_pred)
                if len(eps_history) > 4:
                    eps_history.pop(0)

                # 2. Compute PLMS update using equation (4) from the paper
                if len(eps_history) == 4:
                    # Pseudo-linear multi-step update
                    eps_prime = (55 * eps_history[-1] - 59 * eps_history[-2] 
                            + 37 * eps_history[-3] - 9 * eps_history[-4]) / 24
                else:
                    # Fallback to simple prediction
                    eps_prime = eps_pred

                # 3. Apply numerical stability checks
                with torch.no_grad():
                    # Get alpha values with numerical stability
                    alpha_bar_t = self._process.sqrt_alpha_bar(t_tensor).view(-1, 1)**2
                    alpha_bar_prev = self._process.sqrt_alpha_bar(t_tensor-1).view(-1, 1)**2
                    
                    # Clamp to prevent extreme values
                    alpha_bar_t = torch.clamp(alpha_bar_t, eps, 1.0-eps)
                    alpha_bar_prev = torch.clamp(alpha_bar_prev, eps, 1.0-eps)
                    
                    # Compute coefficients using equation (2)
                    coeff1 = torch.sqrt(alpha_bar_prev / alpha_bar_t)
                    coeff2 = torch.sqrt(1 - alpha_bar_prev) - torch.sqrt((alpha_bar_prev * (1 - alpha_bar_t)) / alpha_bar_t)
                    
                    # Apply clamping to coefficients
                    coeff1 = torch.clamp(coeff1, -clamp_threshold, clamp_threshold)
                    coeff2 = torch.clamp(coeff2, -clamp_threshold, clamp_threshold)
                    
                    # Update x using equation (2)
                    x = coeff1 * x + coeff2 * eps_prime

                # 4. Post-update checks
                if torch.isnan(x).any():
                    raise RuntimeError(f"NaN detected at step {t} (i={i})")
                    
            return x