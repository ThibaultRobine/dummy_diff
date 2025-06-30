# diffusion_postprocessor_mine.py
from typing import Any
import torch
import dataclasses
from openood.postprocessors import BasePostprocessor
from model_unet import DiscreteDDPMProcess, UniformDiscreteTimeSampler, DiffusionModel
from algorithm_mine import (kmeans_x_ref_list, optimize_reference_point,
                          gaussianQuadrature, train_kde)
from diffusion_model_manager_mine import get_trained_diffusion_model, set_trained_diffusion_model
import pytorch_warmup as warmup
from torch import optim
from diffusion_training_mine import build_or_load_diffusion_model

@dataclasses.dataclass
class NetConfig:
    activation: str
    time_embedding_dim: int

class CustomDiffusionPostprocessorTest(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.setup_flag = False
        
        # Get all parameters from config
        post_cfg = config.get('postprocessor', {})
        self.APS_mode = post_cfg.get('APS_mode', False)
        self.diffusion_args = post_cfg.get('diffusion_args', {})
        self.integration_args = post_cfg.get('integration_method', {})

        # State variables
        self.model = None
        self.x_ref = None
        self.kde = None
        self.scaler = None
        self.train_mean = None
        self.train_std = None

    def setup(self, net, id_loader_dict, ood_loader_dict):
        if self.setup_flag:
            return
        self.setup_flag = True

        # 1. Feature extraction and normalization
        feats_list = []
        for batch in id_loader_dict['train']:
            with torch.no_grad():
                _, feats = net(batch['data'].cuda(), return_feature=True)
                feats_list.append(feats.cpu())
        all_feats = torch.cat(feats_list, dim=0).to('cuda')
        
        # Normalization
        self.train_mean = all_feats.mean(dim=0, keepdim=True)
        self.train_std = all_feats.std(dim=0, keepdim=True) + 1e-6
        all_feats = (all_feats - self.train_mean) / self.train_std

        # 2. Model initialization
        # 2. Model initialization
        self.model = build_or_load_diffusion_model(
        diffusion_args = self.diffusion_args,
        feature_bank = all_feats,        
        device         = 'cuda'            
        )
        # shared_model = get_trained_diffusion_model()
        # if shared_model is None:
        #     # Get all parameters from config
        #     diffusion_params = self.diffusion_args
        #     unet_params = diffusion_params.get('unet', {})
        #     net_config = NetConfig(
        #         activation=unet_params.get('activation'),
        #         time_embedding_dim=unet_params.get('time_embedding_dim')
        #     )
            
        #     # Initialize diffusion process
        #     diffusion_process = DiscreteDDPMProcess(
        #         num_diffusion_timesteps=diffusion_params['num_diffusion_timesteps'],
        #         schedule_type=diffusion_params['schedule_type']
        #     )
            
        #     self.model = DiffusionModel(
        #         diffusion_process=diffusion_process,
        #         time_sampler=UniformDiscreteTimeSampler(
        #             diffusion_process.tmin, 
        #             diffusion_process.tmax
        #         ),
        #         net_config=net_config,
        #         data_shape=(all_feats.shape[1],)
        #     ).cuda()

        #     # 3. Training setup from config
        #     train_cfg = diffusion_params['train_loop']
        #     optimizer = optim.Adam(self.model.parameters(), lr=train_cfg['lr'])
            
        #     # LR scheduling
        #     lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #         optimizer, 
        #         T_0=train_cfg['total_steps'],
        #         eta_min=train_cfg.get('eta_min', 1e-12)
        #     )
        #     warmup_scheduler = warmup.LinearWarmup(
        #         optimizer, 
        #         warmup_period=train_cfg['warmup_steps']
        #     )

        #     # 4. Training loop with direct batch sampling
        #     self.model.train()
        #     for step in range(train_cfg['total_steps']):
        #         # Generate random batch directly from features
        #         x0 = self.generate_batch(all_feats, train_cfg['batch_size']).to('cuda')

        #         optimizer.zero_grad()
        #         loss = self.model.loss(x0)
        #         loss.backward()
        #         optimizer.step()
        #         if step % 50 == 0:
        #             print(f'Step: {step}, Loss: {loss:.5f}')

        #         # Learning rate updates
        #         with warmup_scheduler.dampening():
        #             if warmup_scheduler.last_step + 1 >= train_cfg['warmup_steps']:
        #                 lr_scheduler.step()
        #             if warmup_scheduler.last_step + 1 >= train_cfg['max_step']:
        #                 break

        #     self.model.eval()
        #     set_trained_diffusion_model(self.model)
        # else:
        #     self.model = shared_model

        # 5. Reference point optimization
        self.x_ref = optimize_reference_point(
            model=self.model,
            initial_x=kmeans_x_ref_list(all_feats, self.integration_args['kmeans_k']),
            num_steps=self.integration_args['asc_steps'],
            lr=self.integration_args['asc_lr'],
            max_grad_norm=self.integration_args['asc_grad_clip'],
            weight_decay=self.integration_args['asc_wd'],
            device='cuda'
        )

        # 6. KDE training
        scores_list = []
        for x_ref in self.x_ref:
            sc = gaussianQuadrature(
                model=self.model,
                x=all_feats,
                x_ref=x_ref,
                n=self.integration_args['gauss_n'],
                batch_size=self.integration_args['gauss_batch']
            )
            scores_list.append(sc.unsqueeze(1))

        scores_tensor = torch.cat(scores_list, dim=1)
        dists_tensor  = torch.cdist(all_feats, self.x_ref)
        features_for_kde = torch.cat([scores_tensor, dists_tensor], dim=1)
        
        self.kde, self.scaler = train_kde(features_for_kde)

    @torch.no_grad()
    def postprocess(self, net: torch.nn.Module, data: Any):
        logits = net(data)
        pred = logits.argmax(dim=1)
        # Feature extraction and normalization
        _, feats = net(data, return_feature=True)
        feats = (feats - self.train_mean.to(feats.device)) / self.train_std.to(feats.device)

        # Score computation
        all_scores_list = []
        for x_ref in self.x_ref:
            sc = gaussianQuadrature(
                model=self.model,
                x=feats,
                x_ref=x_ref,
                n=self.integration_args['gauss_n'],
                batch_size=self.integration_args['gauss_batch']
            )
            all_scores_list.append(sc.unsqueeze(1))
            
        # Concatenate all scores
        all_scores_list = torch.cat(all_scores_list, dim=1)
        # Compute distances
        dists_tensor = torch.cdist(feats, self.x_ref)
        # Concatenate scores and distances
        features_for_kde = torch.cat([all_scores_list, dists_tensor], dim=1)
        # Scale features
        scaled_scores = self.scaler.transform(features_for_kde.cpu().numpy())

        return (pred, torch.from_numpy(self.kde.score_samples(scaled_scores)).to(feats.device))
    # create a generate batch function using torch
    def generate_batch(self,data,batch_size):
        # use torche's random choice function to sample from latent space
        idx = torch.randint(0, data.shape[0], (batch_size,))
        return data[idx]