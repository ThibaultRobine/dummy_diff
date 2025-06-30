# src/diffusion_postprocessor.py

from typing import Any
import torch, gc
from openood.postprocessors import BasePostprocessor

import improved_diffusion.dist_util as dist_util
from improved_diffusion.train_util import TrainLoop
from improved_diffusion.resample import create_named_schedule_sampler
from algorithm import kmeans_x_ref_list, optimize_reference_point, gaussianQuadrature, train_kde
from custom_wrapper import create_custom_model_and_diffusion
from diffusion_model_manager import get_trained_diffusion_model, set_trained_diffusion_model



def _dbg(tag):
    torch.cuda.empty_cache(); gc.collect()
    print(f"[{tag}] alloc={torch.cuda.memory_allocated()/2**20:.0f} MB | "
          f"reserv={torch.cuda.memory_reserved()/2**20:.0f} MB")

class DiffusionPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        post_cfg = config.get('postprocessor', {})
        self.APS_mode = post_cfg.get('APS_mode', False)
        self.diffusion_args = post_cfg.get('diffusion_args', {})
        self.integration_args = post_cfg.get('integration_method', {})

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
            print("[DEBUG][SETUP] Initial features - ",
              f"shape: {features.shape}, ",
              f"mean: {features.mean().item():.3f}, ",
              f"std: {features.std().item():.3f}")
        net.train()

        feats_list = []
        for batch in id_loader_dict['train']:
            with torch.no_grad():
                _, feats = net(batch['data'].cuda(), return_feature=True)
                feats_list.append(feats.cpu())
        all_feats = torch.cat(feats_list, dim=0).to('cuda')  # [50000, 512]
        
        # Normalize using GLOBAL stats
        self.train_mean = all_feats.mean(dim=0, keepdim=True)  # [1, 512]
        self.train_std = all_feats.std(dim=0, keepdim=True) + 1e-6
        all_feats = (all_feats - self.train_mean) / self.train_std

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
            _dbg("before TrainLoop")
            train_loop_cfg = self.diffusion_args['train_loop']
            schedule_sampler = create_named_schedule_sampler("uniform", diffusion)
            self.train_loop = TrainLoop(
                model=model,
                diffusion=diffusion,
                data=self._make_feature_generator(net,id_loader_dict['train']),
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
            _dbg("after TrainLoop")
            model.eval()
            set_trained_diffusion_model(model, diffusion)

        feats_list = []
        for batch_data in id_loader_dict['train']:
            with torch.no_grad():
                imgs = batch_data['data'].cuda()
                _, feats_b = net(imgs, return_feature=True)
                feats_list.append(feats_b.cpu())
        all_feats = torch.cat(feats_list, dim=0).to('cuda')
        print("[DEBUG][SETUP] All ID features - ",
              f"shape: {all_feats.shape}, ",
              f"mean: {all_feats.mean().item():.3f}, ",
              f"std: {all_feats.std().item():.3f}")


        kmeans_k = self.integration_args['kmeans_k']
        asc_steps = self.integration_args['asc_steps']
        asc_lr = self.integration_args['asc_lr']
        asc_grad_clip = self.integration_args['asc_grad_clip']
        asc_wd = self.integration_args['asc_wd']
        asc_device = self.integration_args['device']
        gauss_n = self.integration_args['gauss_n']
        gauss_batch = self.integration_args['gauss_batch']

        asc_lr_scheduler = self.integration_args['asc_lr_scheduler']            # e.g. None or ("exponential", {...})
        asc_conv_window = self.integration_args['asc_convergence_window']       # e.g. 2000
        asc_conv_thresh = self.integration_args['asc_convergence_threshold']    # e.g. 1e-6
        asc_min_steps = self.integration_args['asc_min_steps']                  # e.g. 100

        init_refs = kmeans_x_ref_list(all_feats, kmeans_k)
        ref_points = optimize_reference_point(
            model=self.diffusion_model,
            diffusion = self.diffusion_obj,
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
                    diffusion = self.diffusion_obj,
                    x=all_feats,
                    x_ref=x_ref ,
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
        feats = (feats - self.train_mean.to(feats.device)) / self.train_std.to(feats.device)
        print("[DEBUG][INFER] Test features - ",
              f"shape: {feats.shape}, ",
              f"mean: {feats.mean().item():.3f}, ",
              f"std: {feats.std().item():.3f}")

        int_cfg = self.integration_args
        gauss_n = int_cfg['gauss_n']
        gauss_batch = int_cfg['gauss_batch']
        asc_device = int_cfg['device']

        all_scores_list = []
        for x_ref in self.x_ref:
            sc = gaussianQuadrature(
                model=self.diffusion_model,
                diffusion = self.diffusion_obj,
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
                    feats = ( feats - self.train_mean) / self.train_std
                    feats = feats.unsqueeze(1)
                    
                yield feats, {}
