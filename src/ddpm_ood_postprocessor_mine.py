# ddpm_ood_postprocessor.py (your existing file)

from typing import Any
import torch
from openood.postprocessors import BasePostprocessor
import dataclasses
from model_unet import DiscreteDDPMProcess, UniformDiscreteTimeSampler, DiffusionModel

from custom_wrapper import create_custom_model_and_diffusion
from diffusion_model_manager_mine import get_trained_diffusion_model, set_trained_diffusion_model
from algorithm_mine import ddpm_ood_reconstruct_1d  
import pytorch_warmup as warmup
from torch import optim
from diffusion_training_mine import build_or_load_diffusion_model


@dataclasses.dataclass
class NetConfig:
    activation: str
    time_embedding_dim: int

class DdpmOODPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        post_cfg = config.get('postprocessor', {})
        self.APS_mode = post_cfg.get('APS_mode', False)
        self.diffusion_args = post_cfg.get('diffusion_args', {})
        self.ddpmood_args = post_cfg.get('ddpmood_args', {})

        self.setup_flag = False
        self.model = None
        self.train_loop = None
        self.config = config

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
        #     if diffusion_params['schedule_type'] == 'adaptive':
        #         time_sampler = AdaptiveTimeSampler(diffusion_process)     # <<<
        #     else:
        #         time_sampler = UniformDiscreteTimeSampler(                # <<<
        #             diffusion_process.tmin, diffusion_process.tmax)
        #     self.model = DiffusionModel(
        #         diffusion_process=diffusion_process,
        #         time_sampler=time_sampler,
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

    @torch.no_grad()
    def postprocess(self, net: torch.nn.Module, data: Any):
        logits = net(data)
        pred = logits.argmax(dim=1)

        _, feats = net(data, return_feature=True)
        feats = (feats - self.train_mean.to(feats.device)) / self.train_std.to(feats.device)

        recon, mse_list = ddpm_ood_reconstruct_1d(
            feats, self.model,
            self.ddpmood_args.get('ood_t'),
            self.ddpmood_args.get('num_inference_steps'),
        )

        ood_score = -torch.tensor(mse_list, device=feats.device, dtype=torch.float)

        return pred, ood_score

    def generate_batch(self,data,batch_size):
        # use torche's random choice function to sample from latent space
        idx = torch.randint(0, data.shape[0], (batch_size,))
        return data[idx]

