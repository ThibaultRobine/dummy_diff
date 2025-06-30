from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm.auto import tqdm
from time import time
import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


# def optimize_reference_point(
#     model,
#     initial_x,  # Shape: [batch_size, latent_dim]
#     num_steps=10000,
#     lr=1e-4,
#     max_grad_norm=1.0,
#     weight_decay=1e-4,
#     lr_scheduler=None,
#     convergence_window=5000,
#     convergence_threshold=1e-2,
#     min_steps=100,
#     device='cuda'
# ):
#     """
#     Batch-optimized score-based gradient ascent for multiple initial points

#     Args:
#         initial_x: Initial starting points (torch.Tensor) of shape [batch_size, latent_dim]
#     """
#     # Ensure input is properly shaped
#     if initial_x.dim() == 1:
#         initial_x = initial_x.unsqueeze(0)

#     x = torch.nn.Parameter(initial_x.clone().to(device))
#     optimizer = torch.optim.Adam([x], lr=lr, weight_decay=weight_decay)

#     # Configure learning rate scheduler
#     if lr_scheduler:
#         scheduler_type, scheduler_kwargs = lr_scheduler
#         if scheduler_type == 'exponential':
#             scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_kwargs)
#         elif scheduler_type == 'step':
#             scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_kwargs)

#     # Convergence tracking (per-sample)
#     batch_size = x.shape[0]
#     score_history = torch.full((num_steps, batch_size), float('nan'), device=device)
#     active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

#     with tqdm(total=num_steps, desc="Batch Score Ascent") as pbar:
#         for step in range(num_steps):
#             optimizer.zero_grad()

#             # Get scores only for active samples
#             with torch.no_grad():
#                 active_x = x[active_mask]
#                 if active_x.numel() == 0:
#                     break

#                 # Get score for active samples
#                 score = -model.net_fwd(active_x, torch.zeros(active_x.size(0), device=device))

#                 # Store scores
#                 score_history[step, active_mask] = score.norm(dim=1)

#             # Manual gradient setup
#             x.grad = torch.zeros_like(x)
#             x.grad[active_mask] = -score  # Apply gradients only to active samples

#             # Gradient clipping
#             if max_grad_norm is not None:
#                 torch.nn.utils.clip_grad_norm_([x], max_grad_norm)

#             optimizer.step()

#             # Update learning rate
#             if lr_scheduler:
#                 scheduler.step()

#             # Check convergence for each sample
#             if step > min_steps:
#                 # Calculate RMS change for each sample
#                 recent_scores = score_history[step-convergence_window:step]
#                 rms_changes = recent_scores.diff(dim=0).pow(2).mean(dim=0).sqrt()

#                 # Update active mask
#                 newly_converged = (rms_changes < convergence_threshold) & active_mask
#                 active_mask[newly_converged] = False

#                 # Update progress bar
#                 pbar.set_postfix({
#                     'active': f"{active_mask.sum().item()}/{batch_size}",
#                     'max_score': f"{score_history[:step+1].max():.2e}",
#                     'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
#                 })

#             pbar.update(1)

#             # Early exit if all converged
#             if not active_mask.any():
#                 break

#     return x.detach()
import torch
from collections import deque
from tqdm.auto import tqdm


def optimize_reference_point(
    model,
    initial_x,                          # shape  [batch, latent_dim]
    num_steps: int            = 10_000,
    lr: float                 = 5e-4,
    max_grad_norm: float      = 10.0,
    weight_decay: float       = 0.0,
    lr_scheduler=None,                  
    convergence_window: int   = 500,
    convergence_threshold: float = 1e-1,
    min_steps: int            = 100,
    device: str               = "cuda",
):
    """
    Score-based gradient ascent with a cosine LR schedule.
    Returns the optimised latent points `x` (same shape as `initial_x`).
    """

    # -------- shape & device --------
    if initial_x.dim() == 1:
        initial_x = initial_x.unsqueeze(0)
    x = torch.nn.Parameter(initial_x.clone().to(device))

    # -------- optimiser & cosine LR --------
    optimiser = torch.optim.Adam([x], lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=num_steps,
        eta_min=1e-5,           # hard-coded final LR
    )

    # -------- convergence bookkeeping --------
    batch_size   = x.shape[0]
    window_score = deque(maxlen=convergence_window + 1)        # CPU
    active_mask  = torch.ones(batch_size, dtype=torch.bool, device=device)

    # -------- main loop --------
    with tqdm(total=num_steps, desc="Batch Score Ascent") as pbar:
        for step in range(num_steps):
            optimiser.zero_grad()

            # forward pass on all points
            score  = -model.net_fwd(x, torch.zeros(x.size(0), device=device))
            norms  = score.norm(dim=1)   # GPU  [batch]
            current_max = norms.max().item()

            # store norms on CPU for the sliding window
            window_score.append(norms.detach().cpu())

            # gradient ascent for active points only
            x.grad = torch.zeros_like(x)
            x.grad[active_mask] = -score[active_mask]

            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_([x], max_grad_norm)

            optimiser.step()
            scheduler.step()             # cosine decay

            # ---------- convergence check ----------
            if (
                step > min_steps
                and len(window_score) == window_score.maxlen
            ):
                w    = torch.stack(tuple(window_score))          # [win, batch]
                rms  = w.diff(dim=0).pow(2).mean(dim=0).sqrt()  # diagnostic

                stayed_low = (w < convergence_threshold).all(dim=0)
                active_mask[stayed_low.to(device)] = False

                pbar.set_postfix(
                    active    = f"{active_mask.sum().item()}/{batch_size}",
                    max_score = f"{current_max:.2e}",
                    max_rms   = f"{rms.max():.2e}",
                    lr        = f"{optimiser.param_groups[0]['lr']:.2e}",
                )

            pbar.update(1)
            if not active_mask.any():
                break

    return x.detach()


def projected_score(model,t, x, y,device='cuda'):
    with torch.no_grad():
        z = x * (1 - t) + y * t
        shape = z.shape
        z = z.reshape(-1, z.shape[-1])
        score = model.net_fwd(z, torch.ones(z.shape[0], device=device))
        score = score.view(shape)
        scalar = torch.sum(score * (y - x), dim=1)
    return scalar

def getGaussLegendrePointsAndWeights(n,device='cuda'):
    if n == 2:
        x = torch.tensor([-0.57735, 0.57735], dtype=torch.float32, device=device)
        w = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    elif n == 3:
        x = torch.tensor([-0.774597, 0.0, 0.774597], dtype=torch.float32, device=device)
        w = torch.tensor([0.555556, 0.888889, 0.555556], dtype=torch.float32, device=device)
    elif n == 4:
        x = torch.tensor([-0.861136, -0.339981, 0.339981, 0.861136], dtype=torch.float32, device=device)
        w = torch.tensor([0.347855, 0.652145, 0.652145, 0.347855], dtype=torch.float32, device=device)
    else:
        x_np, w_np = np.polynomial.legendre.leggauss(n)
        x = torch.tensor(x_np, dtype=torch.float32, device=device)
        w = torch.tensor(w_np, dtype=torch.float32, device=device)
    return x, w

def gaussianQuadrature(model,x, x_ref, n, batch_size=5000,device='cuda'):
    p, w = getGaussLegendrePointsAndWeights(n)
    p = p.view(-1, 1)
    w = w.view(-1, 1)
    result = torch.zeros(x.shape[0], device=device)

    num_batches = (x.shape[0] + batch_size - 1) // batch_size
    batch_iter = tqdm(range(num_batches), desc="Quadrature Batches", leave=False)

    for batch_idx in batch_iter:
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, x.shape[0])
        x_batch = x[start:end]
        batch_result = torch.zeros(x_batch.size(0), device=device)

        point_iter = tqdm(range(n), desc="Quadrature Points", leave=False)
        for i in point_iter:
            t_i = 0.5 * (p[i] + 1)
            score = projected_score(model,t_i, x_batch, x_ref)
            batch_result += w[i] * score
            point_iter.set_postfix({'Point': f"{i+1}/{n}"})

        result[start:end] = -0.5 * batch_result
        batch_iter.set_postfix({'Processed': f"{end}/{x.shape[0]}"})

    return result

def kmeans_x_ref_list(data, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data.cpu())
    return torch.tensor(kmeans.cluster_centers_,dtype=torch.float32)

def train_kde(id_scores):
    """Train KDE ONLY on training split of ID data"""
    with tqdm(total=3, desc="Training KDE") as pbar:
        # Convert to numpy and split ID data into TRAIN/VAL
        id_scores_np = id_scores.cpu().numpy()

        # Normalizer fit ONLY on training data
        scaler = StandardScaler().fit(id_scores_np)
        pbar.update(1)

        # Grid search ONLY on training data
        grid = GridSearchCV(
            KernelDensity(kernel='gaussian'),
            {'bandwidth': np.logspace(-2, 1, 20)},
            cv=5,
            n_jobs=-1
        )
        grid.fit(scaler.transform(id_scores_np))  # Critical: no val data here
        pbar.update(1)

        # Final KDE trained on FULL training data
        kde = grid.best_estimator_
        kde.fit(scaler.transform(id_scores_np))
        pbar.update(1)

        return kde, scaler

def UQ_int_score(training_id_data,test_id_data,ood_data, k, n=20):
    """Modified with explicit ID/OOD separation"""
    with tqdm(total=4, desc="Computing UQ Scores") as main_pbar:
        # ========== PHASE 1: Reference points from ID data only ==========
        main_pbar.set_description("Finding reference points")
        x_init_list = kmeans_x_ref_list(training_id_data, k)  # Cluster ID data only/
        x_ref_list = optimize_reference_point(x_init_list)
        main_pbar.update(1)

        # ========== PHASE 2: Compute ID scores for KDE training ==========
        main_pbar.set_description("Computing ID scores")
        id_scores = []
        for x_ref in x_ref_list:
            scores = gaussianQuadrature(training_id_data, x_ref, n)
            id_scores.append(scores.unsqueeze(1))
        id_scores = torch.cat(id_scores, dim=1)  # Shape [N_id, k]
        main_pbar.update(1)

        # ========== PHASE 3: Train KDE on ID data only ==========
        main_pbar.set_description("Training KDE")
        kde_model, scaler = train_kde(id_scores)
        main_pbar.update(1)

        # ========== PHASE 4: Score ALL data (ID + OOD) ==========
        main_pbar.set_description("Scoring all data")
        data = torch.cat([test_id_data, ood_data], dim=0)
        labels = torch.cat([torch.zeros(test_id_data.shape[0]), torch.ones(ood_data.shape[0])], dim=0)
        all_scores = []
        for x_ref in x_ref_list:
            scores = gaussianQuadrature(data, x_ref, n)
            all_scores.append(scores.unsqueeze(1))
        all_scores = torch.cat(all_scores, dim=1).cpu().numpy()

        # Apply normalization from ID training data
        scaled_scores = scaler.transform(all_scores)
        uq_scores = -kde_model.score_samples(scaled_scores)
        main_pbar.update(1)

    return uq_scores,labels


def compute_diffpath_stats(model, data, n_steps=20, batch_size=512, device='cuda'):
    """Batch-processed statistics computation without DataLoader"""
    stats = []
    process = model._process
    alpha_bars = process._sqrt_alphas_bar.cpu() ** 2

    for i in tqdm(range(0, len(data), batch_size),
                    desc="Processing Batches",
                    unit="batch"):
        batch = data[i:i+batch_size].to(device)
        epsilons = []
        xt = batch.clone()

        timesteps = np.linspace(0, process._num_diffusion_timesteps-1, n_steps, dtype=int)[::-1]
        for t in timesteps:
            t_tensor = torch.full((xt.shape[0],), t, device=device)
            eps_pred = model.net_fwd(xt, t_tensor).clamp(-1e3, 1e3).squeeze(1)
            epsilons.append(eps_pred.cpu())

            alpha_bar_t = alpha_bars[t]
            alpha_bar_prev = alpha_bars[t-1] if t > 0 else 1.0
            xt = torch.sqrt(torch.tensor(alpha_bar_prev)) * (
                xt - torch.sqrt(torch.tensor(1 - alpha_bar_t)) * eps_pred
            ) / torch.sqrt(torch.tensor(alpha_bar_t))

        eps = torch.stack(epsilons).numpy()
        eps_sum = eps.sum((0,2))
        eps_sq = (eps**2).sum((0,2))
        eps_cb = (eps**3).sum((0,2))

        eps_diff = np.diff(eps, axis=0) * process._num_diffusion_timesteps
        deps_sum = eps_diff.sum((0,2))
        deps_sq = (eps_diff**2).sum((0,2))
        deps_cb = (eps_diff**3).sum((0,2))

        stats.append(np.vstack([eps_sum, eps_sq, eps_cb, deps_sum, deps_sq, deps_cb]).T)

    return np.concatenate(stats)


def compute_msma_stats(model, data, n_steps=20, batch_size=1024, device='cuda'):
    """Memory-optimized MSMA statistics computation"""
    stats = []
    process = model._process
    alpha_bars = process._sqrt_alphas_bar.cpu() ** 2
    
    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size),
                     desc="MSMA Stats"):
            batch = data[i:i+batch_size].to(device)
            l2_norms = []
            xt = batch.clone()

            timesteps = np.linspace(0, process._num_diffusion_timesteps-1, n_steps, dtype=int)[::-1]
            
            for t in timesteps:
                t_tensor = torch.full((xt.shape[0],), t, device=device)
                eps_pred = model.net_fwd(xt, t_tensor).squeeze(1)
                l2_norms.append(torch.linalg.norm(eps_pred, dim=1).cpu())
                
                alpha_bar_t = alpha_bars[t]
                alpha_bar_prev = alpha_bars[t-1] if t > 0 else 1.0
                xt = (
                    torch.sqrt(torch.tensor(alpha_bar_prev, device=device)) *
                    (xt - torch.sqrt(torch.tensor(1 - alpha_bar_t, device=device)) * eps_pred)
                ) / torch.sqrt(torch.tensor(alpha_bar_t, device=device))
            
            stats.append(torch.stack(l2_norms, dim=1).numpy())

    return np.concatenate(stats)


def ddpm_ood_reconstruct_1d(feats, model, t, num_inference_steps):
    device = feats.device
    B = feats.shape[0]

    process = model._process
    betas = torch.tensor(process._betas, device=device, dtype=torch.float32)
    alphas_cum = (process._sqrt_alphas_bar ** 2).to(device)
    T = betas.shape[0]
    t = min(t, T - 1)

    alpha_t = alphas_cum[t]
    sqrt_alpha_cum = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_cum = torch.sqrt(1.0 - alpha_t)

    noise = torch.randn_like(feats)
    x_t = sqrt_alpha_cum * feats + sqrt_one_minus_alpha_cum * noise

    step_list = torch.linspace(t, 0, num_inference_steps, dtype=torch.long, device=device).unique_consecutive()
    recon = x_t.clone()

    for i in range(len(step_list) - 1):
        curr_t = int(step_list[i].item())
        t_tensor = torch.full((B,), curr_t, device=device, dtype=torch.long)
        eps = model.net_fwd(recon, t_tensor)
        alpha_cum_curr = alphas_cum[curr_t]
        x0_pred = (recon - torch.sqrt(1.0 - alpha_cum_curr) * eps) / torch.sqrt(alpha_cum_curr)
        next_t = int(step_list[i+1].item())
        alpha_cum_next = alphas_cum[next_t]
        recon = torch.sqrt(alpha_cum_next) * x0_pred + torch.sqrt(1.0 - alpha_cum_next) * eps

    mse_list = []
    if recon.dim() == 3 and recon.shape[1] == 1:
        feats_2d = feats.squeeze(1)
        recon_2d = recon.squeeze(1)
    else:
        feats_2d = feats
        recon_2d = recon

    feats_np = feats_2d.cpu().numpy()
    recon_np = recon_2d.cpu().numpy()

    for b in range(B):
        val = np.mean((feats_np[b] - recon_np[b])**2)
        mse_list.append(val)

    return recon, mse_list



import torch
import torch.nn.functional as F
import numpy as np
def broadcast_timesteps(value: torch.Tensor, ref_tensor: torch.Tensor) -> torch.Tensor:
    """
    Broadcasts a [N]-shaped 'value' to match the shape of 'ref_tensor',
    assuming ref_tensor has shape [N, D] (or [N, ...]).
    """
    while value.dim() < ref_tensor.dim():
        value = value.unsqueeze(-1)  # add trailing dims
    return value.expand_as(ref_tensor)

def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    """
    Average over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, tensor.ndim)))

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two normal distributions:
      KL(N(mean1, var1) || N(mean2, var2))
    Returns a tensor of shape [batch_size, ...].
    """
    # Ensure all are same shape
    # KL = 0.5 * [ (logvar2 - logvar1)
    #              + exp(logvar1 - logvar2)
    #              + (mean1 - mean2)^2 / exp(logvar2)
    #              - 1 ]
    return 0.5 * (
        (logvar2 - logvar1)
        + torch.exp(logvar1 - logvar2)
        + (mean1 - mean2)**2 * torch.exp(-logvar2)
        - 1.0
    )

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of x under a discretized Gaussian.
    This is the same discrete Gaussian used in e.g. the Improved Diffusion code.

    x, means, log_scales should be Tensors of the same shape.

    Returns a tensor of shape [batch_size, ...].
    """
    
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 0.5 / 255.0)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 0.5 / 255.0)
    cdf_min = torch.sigmoid(min_in)

    log_cdf_plus = torch.log(cdf_plus.clamp_min(1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp_min(1e-12))
    cdf_delta = cdf_plus - cdf_min
    mid_in = inv_stdv*centered_x
    log_pdf_mid = mid_in - log_scales - 2.0*F.softplus(mid_in)

    # Use logic from the official code:
    # 1. cdf_delta > 1e-5 => safe to take log
    # 2. else fallback to approximation
    cond1 = (cdf_delta > 1e-5).float()
    log_probs = cond1 * torch.log(cdf_delta.clamp_min(1e-12)) \
                + (1.-cond1) * (log_pdf_mid - np.log(127.5))  # fallback
    if torch.isnan(log_probs).any():
        print("NaN detected: log_probs; count =",
              torch.isnan(log_probs).sum().item())

    # edge cases for x near -1 or 1 can be clipped, but in typical code we rely on them being in [-1,1].
    return log_probs
def q_posterior_mean_variance(model, x_start, x_t, t_batch):
    device = x_start.device
    alpha_bar_t     = (model._process._sqrt_alphas_bar[t_batch])**2
    alpha_bar_tprev = (model._process._sqrt_alphas_bar[(t_batch-1).clamp_min(0)])**2

    betas_t = torch.from_numpy(model._process._betas).to(device=device, dtype=torch.float32)
    beta_t  = betas_t[(t_batch-1).clamp_min(0)]

    denom  = (1. - alpha_bar_t).clamp_min(1e-20)
    numer  = beta_t * (1. - alpha_bar_tprev)
    var_q  = numer / denom
    logvar_q = torch.log(var_q.clamp_min(1e-20))

    sqrt_ab_tprev = torch.sqrt(alpha_bar_tprev)
    sqrt_ab_t     = torch.sqrt(alpha_bar_t)
    c1 = (beta_t * sqrt_ab_tprev) / denom
    c2 = ((1. - alpha_bar_tprev) * sqrt_ab_t) / denom

    # shape => [N,1] times [N, 2048] => [N, 2048]
    posterior_mean = c1.unsqueeze(1)*x_start + c2.unsqueeze(1)*x_t

    # Now broadcast var_q, logvar_q to [N, 2048] for consistent shape
    var_q    = var_q.unsqueeze(1).expand_as(posterior_mean)
    logvar_q = logvar_q.unsqueeze(1).expand_as(posterior_mean)

    return posterior_mean, var_q, logvar_q

def p_mean_variance(model, x_t, t_batch, clip_denoised=True):
    device = x_t.device

    # 1) predict eps
    eps_pred = model.net_fwd(x_t, t_batch)  # shape [N, 2048]

    # 2) alpha_bar_t, etc are shape [N]
    alpha_bar_t = (model._process._sqrt_alphas_bar[t_batch])**2
    sqrt_ab_t   = alpha_bar_t.sqrt().clamp_min(1e-20)
    one_minus_ab_t = (1. - alpha_bar_t).clamp_min(1e-20)

    betas_t = torch.from_numpy(model._process._betas).to(device, torch.float32)
    beta_t  = betas_t[(t_batch-1).clamp_min(0)]

    sigma_t = torch.sqrt(one_minus_ab_t)
    # pred_xstart => shape [N, 2048]
    pred_xstart = (x_t - sigma_t.unsqueeze(1)*eps_pred) / sqrt_ab_t.unsqueeze(1)
    if clip_denoised:
        pred_xstart = torch.clamp(pred_xstart, -1., 1.)

    # alpha_t = alpha_bar_t / alpha_bar_{t-1}, but let's go simpler:
    sqrt_alpha_t = torch.sqrt(
        alpha_bar_t / (
            (model._process._sqrt_alphas_bar[(t_batch-1).clamp_min(0)])**2 + 1e-20
        )
    )
    sqrt_alpha_t = sqrt_alpha_t.clamp_min(1e-20)

    # mean => shape [N, 2048]
    mean = (1./sqrt_alpha_t).unsqueeze(1) * (
        x_t - (beta_t / torch.sqrt(one_minus_ab_t)).unsqueeze(1)*eps_pred
    )

    # var => shape [N], log_var => shape [N]
    var = beta_t.clamp_min(1e-20)
    log_var = torch.log(var)

    # broadcast them to [N, 2048]
    var     = var.unsqueeze(1).expand_as(x_t)
    log_var = log_var.unsqueeze(1).expand_as(x_t)

    return {
        "mean": mean,            # [N, 2048]
        "variance": var,         # [N, 2048]
        "log_variance": log_var, # [N, 2048]
        "pred_xstart": pred_xstart,  # [N, 2048]
    }

def _vb_terms_bpd(model, x_start, x_t, t_batch, clip_denoised=True):
    # True posterior
    true_mean, true_var, true_log_var = q_posterior_mean_variance(model, x_start, x_t, t_batch)

    # Model's distribution
    out = p_mean_variance(model, x_t, t_batch, clip_denoised=clip_denoised)

    model_mean, model_log_var = out["mean"], out["log_variance"]

    # Now all four are shape [N, 2048], so normal_kl won't fail
    kl = normal_kl(true_mean, true_log_var, model_mean, model_log_var)
    kl = mean_flat(kl) / np.log(2.0)   # => shape [N]

    # decoder NLL
    decoder_nll = -discretized_gaussian_log_likelihood(
        x_start,
        means=out["mean"],               # [N, 2048]
        log_scales=0.5*out["log_variance"]  # also [N, 2048]
    )
    decoder_nll = mean_flat(decoder_nll) / np.log(2.0)  # => shape [N]

    # pick kl vs decoder_nll
    use_nll_mask = (t_batch == 0).float()  # shape [N]
    output = use_nll_mask * decoder_nll + (1.-use_nll_mask)*kl  # => [N]
    # --- NaN check -----------------------------------------------------
    if torch.isnan(output).any():
        print("NaN detected in 'output' at t =", int(t_batch[0]),
              "count =", torch.isnan(output).sum().item())
    # ------------------------------------------------------------------

    return {
        "output": output,            # [N]
        "pred_xstart": out["pred_xstart"],  # [N, 2048]
    }

def _prior_bpd(model, x_start):
    """
    The prior KL term in bits-per-dim:
      KL(q(x_T|x_0) || N(0,I)).
    """
    T = model._process._num_diffusion_timesteps
    device = x_start.device

    # shape [N]
    t_batch = torch.full((x_start.shape[0],), T, device=device, dtype=torch.long)

    # alpha_bar_T => shape [N]
    alpha_bar_T = (model._process._sqrt_alphas_bar[t_batch])**2

    # mu_q => shape [N, 2048]
    mu_q = torch.sqrt(alpha_bar_T).unsqueeze(1)* x_start

    # var_q => shape [N], broadcast to [N, 2048]
    var_q = (1. - alpha_bar_T).clamp_min(1e-20)
    var_q = var_q.unsqueeze(1).expand_as(mu_q)     # => [N, 2048]

    # logvar_q => same shape [N, 2048]
    logvar_q = torch.log(var_q)

    # for p(x_T) = N(0, I), we want mean2=0, logvar2=0 with shape [N, 2048]
    mean2   = torch.zeros_like(mu_q)    # => [N, 2048]
    logvar2 = torch.zeros_like(mu_q)    # => [N, 2048]

    kl = normal_kl(
        mean1=mu_q,
        logvar1=logvar_q,
        mean2=mean2,
        logvar2=logvar2,
    )
    # average over dims [1..] => get per-batch
    kl = mean_flat(kl) / np.log(2.0)    # => shape [N]
    return kl

import torch

def calc_bpd_loop(model, x_start, clip_denoised=True):
    """
    Compute the entire variational lower-bound in bits-per-dim, plus
    intermediate arrays (vb, xstart_mse, mse) at each diffusion step.

    :param model: your DiffusionModel containing:
        - model._process (the forward process)
        - net_fwd (the model predicting eps)
    :param x_start: [N x D] or [N x ...] tensor of the original data.
    :param clip_denoised: if True, clip the predicted x_0 to [-1,1].
    :return: a dict:
      {
        'total_bpd': [N]-shaped,
        'prior_bpd': [N]-shaped,
        'vb': [N x T],
        'xstart_mse': [N x T],
        'mse': [N x T]
      }
    """
    device = x_start.device
    batch_size = x_start.shape[0]
    T = model._process._num_diffusion_timesteps

    vb_terms = []
    xstart_mse_terms = []
    eps_mse_terms = []

    for t in reversed(range(1, T + 1)):
        # t goes T-1, T-2, ..., 0
        t_batch = torch.tensor([t]*batch_size, device=device, dtype=torch.long)

        # 1) Sample x_t from the forward process q(x_t|x_0)
        noise = torch.randn_like(x_start)
        x_t = model._process.sample(x0=x_start, t=t_batch, eps=noise)

        # 2) Compute the VLB term at this timestep
        with torch.no_grad():
            out = _vb_terms_bpd(
                model,
                x_start=x_start,
                x_t=x_t,
                t_batch=t_batch,
                clip_denoised=clip_denoised
            )
        vb_terms.append(out["output"])  # shape [N]

        # 3) MSE of predicted x_0
        pred_x0 = out["pred_xstart"]
        xstart_mse_terms.append( mean_flat( (pred_x0 - x_start)**2 ) )  # shape [N]

        # 4) MSE of predicted eps
        #    Re-derive predicted eps as:  eps_hat = (x_t - sqrt_ab_t * x0_pred) / sigma_t
        alpha_bar_t   = (model._process._sqrt_alphas_bar[t_batch])**2
        sigma_t       = torch.sqrt( (1. - alpha_bar_t).clamp_min(1e-20) )
        eps_hat       = (x_t - pred_x0*torch.sqrt(alpha_bar_t).view(-1,1)) / sigma_t.view(-1,1)
        eps_mse_terms.append( mean_flat( (eps_hat - noise)**2 ) )  # shape [N]

    # Stack them in time dimension: result is [N x T].
    vb = torch.stack(vb_terms, dim=1)
    xstart_mse = torch.stack(xstart_mse_terms, dim=1)
    mse = torch.stack(eps_mse_terms, dim=1)

    # The prior bpd
    prior_bpd = _prior_bpd(model, x_start)  # shape [N]

    # Summation over timesteps
    total_bpd = vb.sum(dim=1) + prior_bpd  # shape [N]

    return {
        "total_bpd": total_bpd,
        "prior_bpd": prior_bpd,
        "vb": vb,
        "xstart_mse": xstart_mse,
        "mse": mse,
    }


import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def compute_nll_bpd(model, data, batch_size=2048, device='cuda'):
    model.eval()
    scores = []

    for start_idx in tqdm(range(0, len(data), batch_size), desc="Computing BPD", unit="batch"):
        end_idx = start_idx + batch_size
        batch = data[start_idx:end_idx].to(device)

        with torch.no_grad():
            results = calc_bpd_loop(model, batch, clip_denoised=True)
            bpd_vals = results["total_bpd"]
            # --- NaN check -------------------------------------------------
            nan_cnt = torch.isnan(bpd_vals).sum().item()
            if nan_cnt:
                print(f"NaN detected in batch {start_idx//batch_size}, count =", nan_cnt)
            # ---------------------------------------------------------------

        scores.append(bpd_vals)

    return torch.cat(scores, dim=0).to(device)
