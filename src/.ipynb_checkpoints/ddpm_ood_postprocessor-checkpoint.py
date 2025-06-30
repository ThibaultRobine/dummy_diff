# ddpm_ood_postprocessor.py (your existing file)

from typing import Any
import torch
from openood.postprocessors import BasePostprocessor

import improved_diffusion.dist_util as dist_util
from improved_diffusion.train_util import TrainLoop
from improved_diffusion.resample import create_named_schedule_sampler
from custom_wrapper import create_custom_model_and_diffusion
from diffusion_model_manager import get_trained_diffusion_model, set_trained_diffusion_model
from algorithm import ddpm_ood_reconstruct_1d  


class DdpmOODPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        post_cfg = config.get('postprocessor', {})
        self.APS_mode = post_cfg.get('APS_mode', False)
        self.diffusion_args = post_cfg.get('diffusion_args', {})
        self.ddpmood_args = post_cfg.get('ddpmood_args', {})

        self.setup_flag = False
        self.diffusion_model = None
        self.diffusion_obj = None
        self.train_loop = None
        self.config = config

    def setup(self, net, id_loader_dict, ood_loader_dict):
        if self.setup_flag:
            return
        self.setup_flag = True

        dist_util.setup_dist()

        first_batch = next(iter(id_loader_dict['train']))
        images = first_batch['data'].cuda()
        net.eval()
        with torch.no_grad():
            _, features = net(images, return_feature=True)
        net.train()

        if features.dim() != 2:
            raise RuntimeError(f"Expected penultimate features to be [B, D], got {features.shape}!")
        length_d = features.shape[1]

        dims = self.diffusion_args['dims']
        in_channels = self.diffusion_args['in_channels']
        out_channels = self.diffusion_args['out_channels']
        model_channels = self.diffusion_args['model_channels']
        channel_mult = self.diffusion_args['channel_mult']
        num_res_blocks = self.diffusion_args['num_res_blocks']
        attention_resolutions = self.diffusion_args['attention_resolutions']
        dropout = self.diffusion_args['dropout']
        class_cond = self.diffusion_args.get('class_cond', False)
        diffusion_steps = self.diffusion_args['diffusion_steps']
        noise_schedule = self.diffusion_args['noise_schedule']
        learn_sigma = self.diffusion_args.get('learn_sigma', False)
        sigma_small = self.diffusion_args.get('sigma_small', False)
        predict_xstart = self.diffusion_args.get('predict_xstart', False)
        rescale_timesteps = self.diffusion_args.get('rescale_timesteps', True)
        rescale_learned_sigmas = self.diffusion_args.get('rescale_learned_sigmas', True)
        use_kl = self.diffusion_args.get('use_kl', False)
        timestep_respacing = self.diffusion_args.get('timestep_respacing', "")
        num_heads = self.diffusion_args.get('num_heads', 1)
        num_heads_upsample = self.diffusion_args.get('num_heads_upsample', -1)
        use_checkpoint = self.diffusion_args.get('use_checkpoint', False)
        use_scale_shift_norm = self.diffusion_args.get('use_scale_shift_norm', True)

        image_size = length_d
        shared_diffusion_model = get_trained_diffusion_model()
        if shared_diffusion_model is not None:
            model, diffusion = shared_diffusion_model
            model.cuda()
            model.eval()
            self.diffusion_model = model
            self.diffusion_obj = diffusion
        else:
            model, diffusion = create_custom_model_and_diffusion(
                in_channels=in_channels,
                out_channels=out_channels,
                dims=dims,
                image_size=image_size,
                model_channels=model_channels,
                channel_mult=channel_mult,
                num_res_blocks=num_res_blocks,
                attention_resolutions=attention_resolutions,
                dropout=dropout,
                class_cond=class_cond,
                diffusion_steps=diffusion_steps,
                noise_schedule=noise_schedule,
                learn_sigma=learn_sigma,
                sigma_small=sigma_small,
                predict_xstart=predict_xstart,
                rescale_timesteps=rescale_timesteps,
                rescale_learned_sigmas=rescale_learned_sigmas,
                use_kl=use_kl,
                timestep_respacing=timestep_respacing,
                num_heads=num_heads,
                num_heads_upsample=num_heads_upsample,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm
            )
            model.cuda()
            model.train()
            self.diffusion_model = model
            self.diffusion_obj = diffusion

            train_loop_cfg = self.diffusion_args['train_loop']
            schedule_sampler = create_named_schedule_sampler("uniform", diffusion)
            self.train_loop = TrainLoop(
                model=model,
                diffusion=diffusion,
                data=self._make_feature_generator(net, id_loader_dict['train']),
                batch_size=train_loop_cfg['batch_size'],
                microbatch=train_loop_cfg['microbatch'],
                lr=train_loop_cfg['lr'],
                ema_rate=train_loop_cfg['ema_rate'],
                log_interval=train_loop_cfg['log_interval'],
                save_interval=train_loop_cfg['save_interval'],
                resume_checkpoint=train_loop_cfg['resume_checkpoint'],
                use_fp16=train_loop_cfg['use_fp16'],
                fp16_scale_growth=train_loop_cfg['fp16_scale_growth'],
                schedule_sampler=schedule_sampler,
                weight_decay=train_loop_cfg['weight_decay'],
                lr_anneal_steps=train_loop_cfg['lr_anneal_steps']
            )
            self.train_loop.run_loop()
            model.eval()
            set_trained_diffusion_model(model, diffusion)

    @torch.no_grad()
    def postprocess(self, net: torch.nn.Module, data: Any):
        logits = net(data)
        pred = logits.argmax(dim=1)

        _, feats = net(data, return_feature=True)

        recon, mse_list = ddpm_ood_reconstruct_1d(
            feats, self.diffusion_model, self.diffusion_obj,
            self.ddpmood_args.get('ood_t'),
            self.ddpmood_args.get('num_inference_steps'),
        )

        ood_score = torch.tensor(mse_list, device=feats.device, dtype=torch.float)

        return pred, ood_score

    def _make_feature_generator(self, net, loader):
        while True:
            for batch_data in loader:
                images = batch_data['data'].cuda()
                with torch.no_grad():
                    _, feats = net(images, return_feature=True)
                    feats = feats.unsqueeze(1)
                yield feats, {}

