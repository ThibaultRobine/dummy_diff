# src/diffusion_postprocessor.py

from typing import Any
import torch
from openood.postprocessors import BasePostprocessor

import improved_diffusion.dist_util as dist_util
from improved_diffusion.train_util import TrainLoop
from improved_diffusion.resample import create_named_schedule_sampler
from algorithm import kmeans_x_ref_list, optimize_reference_point, gaussianQuadrature, train_kde
from custom_wrapper import create_custom_model_and_diffusion

class DiffusionPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        post_cfg = config.get('postprocessor', {})
        self.APS_mode = post_cfg.get('APS_mode', False)
        self.diffusion_args = post_cfg.get('diffusion_args', {})
        if not self.diffusion_args:
            raise ValueError("Must specify 'diffusion_args' in the config!")

        self.setup_flag = False
        self.diffusion_model = None
        self.diffusion_obj = None
        self.train_loop = None
        self.x_ref = None
        self.kde = None
        self.scaler = None
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

        feats_list = []
        for batch_data in id_loader_dict['train']:
            with torch.no_grad():
                imgs = batch_data['data'].cuda()
                _, feats_b = net(imgs, return_feature=True)
                feats_list.append(feats_b.cpu())
        all_id_features = torch.cat(feats_list, dim=0).to('cuda')

        int_cfg = self.diffusion_args['integration_method']
        kmeans_k = int_cfg['kmeans_k']
        asc_steps = int_cfg['asc_steps']
        asc_lr = int_cfg['asc_lr']
        asc_grad_clip = int_cfg['asc_grad_clip']
        asc_wd = int_cfg['asc_wd']
        asc_device = int_cfg['device']
        gauss_n = int_cfg['gauss_n']
        gauss_batch = int_cfg['gauss_batch']

        # Missing hyperparams from algorithm:
        asc_lr_scheduler = int_cfg['asc_lr_scheduler']            # e.g. None or ("exponential", {...})
        asc_conv_window = int_cfg['asc_convergence_window']       # e.g. 2000
        asc_conv_thresh = int_cfg['asc_convergence_threshold']    # e.g. 1e-6
        asc_min_steps = int_cfg['asc_min_steps']                  # e.g. 100

        init_refs = kmeans_x_ref_list(all_id_features, kmeans_k)
        ref_points = optimize_reference_point(
            model=self.diffusion_model,
            initial_x=init_refs,
            num_steps=asc_steps,
            lr=asc_lr,
            max_grad_norm=asc_grad_clip,
            weight_decay=asc_wd,
            lr_scheduler=asc_lr_scheduler,
            convergence_window=asc_conv_window,
            convergence_threshold=asc_conv_thresh,
            min_steps=asc_min_steps,
            device=asc_device
        )
        self.x_ref = ref_points.to(asc_device)

        scores_list = []
        with torch.no_grad():
            for x_ref in ref_points:
                sc = gaussianQuadrature(
                    model=self.diffusion_model,
                    x=all_id_features,
                    x_ref=x_ref,
                    n=gauss_n,
                    device=asc_device,
                    batch_size=gauss_batch
                )
                scores_list.append(sc.unsqueeze(1))
        id_scores = torch.cat(scores_list, dim=1)

        kde, scaler = train_kde(id_scores)
        self.kde = kde
        self.scaler = scaler

    @torch.no_grad()
    def postprocess(self, net: torch.nn.Module, data: Any):
        logits = net(data)
        pred = logits.argmax(dim=1)

        _, feats = net(data, return_feature=True)

        int_cfg = self.diffusion_args['integration_method']
        gauss_n = int_cfg['gauss_n']
        gauss_batch = int_cfg['gauss_batch']
        asc_device = int_cfg['device']

        all_scores_list = []
        for x_ref in self.x_ref:
            sc = gaussianQuadrature(
                model=self.diffusion_model,
                x=feats,
                x_ref=x_ref,
                n=gauss_n,
                device=asc_device,
                batch_size=gauss_batch
            )
            all_scores_list.append(sc.unsqueeze(1))

        all_scores = torch.cat(all_scores_list, dim=1).cpu().numpy()
        scaled_scores = self.scaler.transform(all_scores)
        kde_scores = -self.kde.score_samples(scaled_scores)
        ood_score = torch.from_numpy(kde_scores).to(logits.device)

        return pred, ood_score

    def _make_feature_generator(self, net, loader):
        while True:
            for batch_data in loader:
                images = batch_data['data'].cuda()
                with torch.no_grad():
                    _, feats = net(images, return_feature=True)
                    feats = feats.unsqueeze(1)
                yield feats, {}
