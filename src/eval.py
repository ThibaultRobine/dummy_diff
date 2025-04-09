# src/eval.py
import torch
import pandas as pd
from openood.evaluation_api import Evaluator
from openood.networks import ResNet18_32x32
from diffusion_postprocessor import DiffusionPostprocessor

ID_NAME = 'cifar10'
DATA_ROOT = './data'

def main():
    net = ResNet18_32x32(num_classes=10)
    net.load_state_dict(torch.load(
        './pretrained_models/cifar10_res18_v1.5/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt'
    ))
    net.cuda().eval()

    config = {
        'postprocessor': {
            'APS_mode': False,
            'diffusion_args': {
                'dims': 1,
                'in_channels': 1,
                'out_channels': 1,
                'model_channels': 64,
                'channel_mult': '1,2,4,8',
                'num_res_blocks': 2,
                'attention_resolutions': '',
                'dropout': 0.0,
                'class_cond': False,
                'diffusion_steps': 1000,
                'noise_schedule': 'linear',
                'learn_sigma': False,
                'sigma_small': False,
                'predict_xstart': False,
                'rescale_timesteps': True,
                'rescale_learned_sigmas': True,
                'use_kl': False,
                'timestep_respacing': '',
                'num_heads': 1,
                'num_heads_upsample': -1,
                'use_checkpoint': False,
                'use_scale_shift_norm': True,

                'train_loop': {
                    'batch_size': 16,
                    'microbatch': 16,
                    'lr': 1e-4,
                    'ema_rate': '0.9999',
                    'log_interval': 1,
                    'save_interval': 1,
                    'resume_checkpoint': None,
                    'use_fp16': False,
                    'fp16_scale_growth': 1.0,
                    'weight_decay': 0.0,
                    'lr_anneal_steps': 1,
                },

                'integration_method': {
                    'kmeans_k': 10,
                    'asc_steps': 1,#10000,
                    'asc_lr': 1e-4,
                    'asc_grad_clip': 1.0,
                    'asc_wd': 1e-4,
                    'device': 'cuda',
                    'gauss_n': 2,#20,
                    'gauss_batch': 200,#2000,

                    'asc_lr_scheduler': None,           
                    'asc_convergence_window': 2000,
                    'asc_convergence_threshold': 1e-6,
                    'asc_min_steps': 100,
                }
            }
        }
    }

    custom_postprocessor = DiffusionPostprocessor(config)
    print("Running CUSTOM evaluation...")
    custom_evaluator = Evaluator(
        net,
        id_name=ID_NAME,
        data_root=DATA_ROOT,
        postprocessor=custom_postprocessor,
        batch_size=200,
        config_root=None
    )
    custom_metrics = custom_evaluator.eval_ood(fsood=False)

    print("Running BASE evaluation...")
    base_evaluator = Evaluator(
        net,
        id_name=ID_NAME,
        data_root=DATA_ROOT,
        postprocessor_name='msp',
        batch_size=200,
        config_root=None
    )
    base_metrics = base_evaluator.eval_ood(fsood=False)

    full_results = pd.concat([custom_metrics, base_metrics],
                             keys=['Custom', 'Base']
    ).reset_index(level=1).rename(columns={'level_1': 'Dataset'})
    full_results.to_csv('full_ood_results.csv')
    print(full_results)

if __name__ == '__main__':
    main()

