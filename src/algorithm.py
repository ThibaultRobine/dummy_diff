import torch
import numpy as np
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

def optimize_reference_point(
    model,
    initial_x,
    num_steps=10000,
    lr=1e-4,
    max_grad_norm=1.0,
    weight_decay=1e-4,
    lr_scheduler=None,
    convergence_window=2000,
    convergence_threshold=1e-6,
    min_steps=100,
    device='cuda'
):
    if initial_x.dim() == 1:
        initial_x = initial_x.unsqueeze(0)
    x = torch.nn.Parameter(initial_x.clone().to(device))
    optimizer = torch.optim.Adam([x], lr=lr, weight_decay=weight_decay)
    if lr_scheduler:
        scheduler_type, scheduler_kwargs = lr_scheduler
        if scheduler_type == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_kwargs)
        elif scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_kwargs)

    batch_size = x.shape[0]
    score_history = torch.full((num_steps, batch_size), float('nan'), device=device)
    active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

    with tqdm(total=num_steps, desc="Batch Score Ascent") as pbar:
        for step in range(num_steps):
            optimizer.zero_grad()
            with torch.no_grad():
                active_x = x[active_mask]
                if active_x.numel() == 0:
                    break
                # Reshape [B, D] -> [B, 1, D] for 1D UNet
                active_x_reshaped = active_x.unsqueeze(1)
                score_reshaped = model(active_x_reshaped, torch.zeros(active_x.size(0), dtype=torch.long, device=device))
                # Now squeeze(1) back to [B, D] so .norm(dim=1) works
                score = score_reshaped.squeeze(1)
                score_history[step, active_mask] = score.norm(dim=1)

            x.grad = torch.zeros_like(x)
            # Reuse score_reshaped (i.e. shape [B,1,D]) if needed, but we'll just do
            x.grad[active_mask] = -score  # shape [B, D]
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_([x], max_grad_norm)
            optimizer.step()
            if lr_scheduler:
                scheduler.step()

            if step > min_steps:
                recent_scores = score_history[step - convergence_window : step]
                rms_changes = recent_scores.diff(dim=0).pow(2).mean(dim=0).sqrt()
                newly_converged = (rms_changes < convergence_threshold) & active_mask
                active_mask[newly_converged] = False
                pbar.set_postfix({
                    'active': f"{active_mask.sum().item()}/{batch_size}",
                    'max_score': f"{score_history[:step+1].max():.2e}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
                    'rms_change': f"{rms_changes.max():.2e}",
                })

            pbar.update(1)
            if not active_mask.any():
                break

    return x.detach()

def projected_score(model, t, x, y, device='cuda'):
    with torch.no_grad():
        z = x * (1 - t) + y * t
        shape = z.shape
        z = z.reshape(-1, z.shape[-1])
        # Reshape to [B,1,D] for model
        z_reshaped = z.unsqueeze(1)
        s_reshaped = model(z_reshaped, torch.zeros(z_reshaped.size(0), dtype=torch.long, device=device))
        # Squeeze back to [B,D]
        s = s_reshaped.squeeze(1)
        s = s.view(shape)
        return torch.sum(s * (y - x), dim=1)

def getGaussLegendrePointsAndWeights(n, device='cuda'):
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

def gaussianQuadrature(model, x, x_ref, n, device='cuda', batch_size=5000):
    p, w = getGaussLegendrePointsAndWeights(n, device=device)
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
            sc = projected_score(model, t_i, x_batch, x_ref, device=device)
            batch_result += w[i] * sc
        result[start:end] = -0.5 * batch_result
    return result

def kmeans_x_ref_list(data, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data.cpu())
    return torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

def train_kde(id_scores):
    id_scores_np = id_scores.cpu().numpy()
    scaler = StandardScaler().fit(id_scores_np)
    grid = GridSearchCV(
        KernelDensity(kernel='gaussian'),
        {'bandwidth': np.logspace(-2, 1, 20)},
        cv=5,
        n_jobs=-1
    )
    grid.fit(scaler.transform(id_scores_np))
    kde = grid.best_estimator_
    kde.fit(scaler.transform(id_scores_np))
    return kde, scaler
