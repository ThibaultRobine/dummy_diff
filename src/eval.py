# src/eval.py
import torch
import pandas as pd
from openood.evaluation_api import Evaluator
from openood.networks import ResNet18_32x32
from diffusion_postprocessor import DiffusionPostprocessor
from diffpath_postprocessor import DiffPathPostprocessor
from diffusion_nll_postprocessor import DiffusionNllPostprocessor
from msma_postprocessor import MSMAPostprocessor
from ddpm_ood_postprocessor import DdpmOODPostprocessor
from diffusion_model_manager import get_trained_diffusion_model, set_trained_diffusion_model, CKPT_DIR
from custom_wrapper           import create_custom_model_and_diffusion

ID_NAME = 'cifar10'
DATA_ROOT = './data'
import gc, os
def show_mem(tag):
    torch.cuda.empty_cache(); gc.collect()
    alloc = torch.cuda.memory_allocated() / 2**20   # in MB
    reserv = torch.cuda.memory_reserved() / 2**20
    print(f"[{tag}] GPU mem  allocated={alloc:>6.0f} MB | "
          f"reserved={reserv:>6.0f} MB")
def _load_latest_diffusion():
    """Look inside trained_diffusion_ckpts/, pick the newest *.pt,
    rebuild the UNet+diffusion, load weights, and cache it."""
    ckpts = sorted(CKPT_DIR.glob("diffusion_*.pt"))
    if not ckpts:
        print("[Diffusion] no previous checkpoint – will train from scratch")
        return                      # nothing to load

    ckpt_path = ckpts[-1]           # newest timestamp
    print(f"[Diffusion] loading checkpoint {ckpt_path.name}")

    # -------------------------
    # 1.  Rebuild model & diffusion with *same* hyper-params
    #     (here we hard-code the ones in your config; you can also
    #     read ckpt['meta'] if you saved more info there)
    # -------------------------
    model, diffusion = create_custom_model_and_diffusion(
        in_channels=1,
        out_channels=1,
        dims=1,
        image_size=128,              # ← length after all downsamples
        model_channels=64,
        channel_mult='1,2,4,8',
        num_res_blocks=2,
        attention_resolutions='',
        dropout=0.0,
        class_cond=False,
        diffusion_steps=1000,
        noise_schedule='linear',
        learn_sigma=False,
        sigma_small=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        use_kl=False,
        timestep_respacing='',
        num_heads=1,
        num_heads_upsample=-1,
        use_checkpoint=False,
        use_scale_shift_norm=True,
    )

    # -------------------------
    # 2.  Load weights
    # -------------------------
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state["state_dict"], strict=True)
    model.cuda().eval()

    # -------------------------
    # 3.  Register so post-processors will reuse it
    # -------------------------
    set_trained_diffusion_model(model, diffusion, ckpt_path)

def main():
    net = ResNet18_32x32(num_classes=10)
    net.load_state_dict(torch.load(
        './pretrained_models/cifar10_res18_v1.5/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt'
    ))
    net.cuda().eval()
    show_mem("after net.cuda")

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
                'linear_start': 0.0001,
                'linear_end': 0.02,
                'learn_sigma': False,
                'sigma_small': False,
                'predict_xstart': False,
                'rescale_timesteps': False,
                'rescale_learned_sigmas': False,
                'use_kl': False,
                'timestep_respacing': '',
                'num_heads': 1,
                'num_heads_upsample': -1,
                'use_checkpoint': False,
                'use_scale_shift_norm': True,
                'loss_type': 'mse',

                'train_loop': {
                    'batch_size': 4096,
                    'microbatch': 512,
                    'lr': 3e-4,
                    'ema_rate': '0',
                    'log_interval': 1,
                    'save_interval': 1,
                    'resume_checkpoint': None,
                    'use_fp16': False,
                    'fp16_scale_growth': 1.0,
                    'weight_decay': 0.0,
                    'lr_anneal_steps': 5000,
                    'save_interval': 5000,
                }
            },
            'integration_method': {
                'kmeans_k': 10,
                'asc_steps': 10000, #10000,
                'asc_lr': 1e-4,
                'asc_grad_clip': 1.0,
                'asc_wd': 1e-4,
                'device': 'cuda',
                'gauss_n': 20,#20,
                'gauss_batch': 6000,#2000,

                'asc_lr_scheduler': None,           
                'asc_convergence_window': 5000,
                'asc_convergence_threshold': 1e-2,
                'asc_min_steps': 100,
            },
            'diffpath_args': {
                'n_steps': 20,#20,
                'batch_size': 2048,
                'device': 'cuda',
            },
            'msma_args': {
                'n_steps': 20,#20,
                'batch_size': 2048,
                'device': 'cuda',
            },
            'ddpm_ood_args': {
                'ood_t' : 250,
                'num_inference_steps' : 100,
            },
        }
    }

    _load_latest_diffusion()

    custom_postprocessor = DiffusionPostprocessor(config)
    show_mem("after postprocessor")
    print("Running OURS evaluation...")
    custom_evaluator = Evaluator(
        net,
        id_name=ID_NAME,
        data_root=DATA_ROOT,
        postprocessor=custom_postprocessor,
        batch_size=2048,
        config_root=None
    )
    show_mem("after evaluator build")
    custom_metrics = custom_evaluator.eval_ood(fsood=False)
    print('OURS metrics:')
    print(custom_metrics)
    results_df = pd.DataFrame(custom_metrics)
    results_df.to_csv('ood_results_ours.csv', index=False)
    print("Results saved to ood_results_ours.csv")

    ddpm_ood_postprocessor = DdpmOODPostprocessor(config)
    print("Running DDPM evaluation...")
    ddpm_evaluator = Evaluator(
        net,
        id_name=ID_NAME,
        data_root=DATA_ROOT,
        postprocessor=ddpm_ood_postprocessor,
        batch_size=2048,
        config_root=None
    )
    ddpm_metrics = ddpm_evaluator.eval_ood(fsood=False)
    print('DDPM metrics:')
    print(ddpm_metrics)
    results_df = pd.DataFrame(ddpm_metrics)
    results_df.to_csv('ood_results_ddpm.csv', index=False)
    print("Results saved to ood_results_ddpm.csv")



    msma_postprocessor = MSMAPostprocessor(config)
    print("Running MSMA evaluation...")
    msma_evaluator = Evaluator(
        net,
        id_name=ID_NAME,
        data_root=DATA_ROOT,
        postprocessor=msma_postprocessor,
        batch_size=2048,
        config_root=None
    )
    msma_metrics = msma_evaluator.eval_ood(fsood=False)
    print('MSMA metrics:')
    print(msma_metrics)
    results_df = pd.DataFrame(msma_metrics)
    results_df.to_csv('ood_results_msma.csv', index=False)
    print("Results saved to ood_results_msma.csv")

    diffnll_postprocessor = DiffusionNllPostprocessor(config)
    print("Running DIFFNLL evaluation...")
    diffnll_evaluator = Evaluator(
        net,
        id_name=ID_NAME,
        data_root=DATA_ROOT,
        postprocessor=diffnll_postprocessor,
        batch_size=2048,
        config_root=None
    )
    diffnll_metrics = diffnll_evaluator.eval_ood(fsood=False)
    print('DIFFNLL metrics:')
    print(diffnll_metrics)
    results_df = pd.DataFrame(diffnll_metrics)
    results_df.to_csv('ood_results_diffnll.csv', index=False)
    print("Results saved to ood_results_diffnll.csv")


    diffpath_postprocessor = DiffPathPostprocessor(config)
    print("Running DIFFPATH evaluation...")
    diffpath_evaluator = Evaluator(
        net,
        id_name=ID_NAME,
        data_root=DATA_ROOT,
        postprocessor=diffpath_postprocessor,
        batch_size=2048,
        config_root=None
    )
    diffpath_metrics = diffpath_evaluator.eval_ood(fsood=False)
    print('DIFFPATH metrics:')
    print(diffpath_metrics)
    results_df = pd.DataFrame(diffpath_metrics)
    results_df.to_csv('ood_results_diffpath.csv', index=False)
    print("Results saved to ood_results_diffpath.csv")


    print("Running ODIN evaluation...")
    odin_evaluator = Evaluator(
        net,
        id_name=ID_NAME,
        data_root=DATA_ROOT,
        postprocessor_name='odin',
        batch_size=2048,
        config_root=None
    )
    odin_metrics = odin_evaluator.eval_ood(fsood=False)
    print('ODIN metrics:')
    print(odin_metrics)
    results_df = pd.DataFrame(odin_metrics)
    results_df.to_csv('ood_results_odin.csv', index=False)
    print("Results saved to ood_results_odin.csv")


    print("Running EBO evaluation...")
    ebo_evaluator = Evaluator(
        net,
        id_name=ID_NAME,
        data_root=DATA_ROOT,
        postprocessor_name='ebo',
        batch_size=2048,
        config_root=None
    )
    ebo_metrics = ebo_evaluator.eval_ood(fsood=False)
    print('EBO metrics:')
    print(ebo_metrics)
    results_df = pd.DataFrame(ebo_metrics)
    results_df.to_csv('ood_results_ebo.csv', index=False)
    print("Results saved to ood_results_ebo.csv")

    print("Running ReAct evaluation...")
    react_evaluator = Evaluator(
        net,
        id_name=ID_NAME,
        data_root=DATA_ROOT,
        postprocessor_name='react',
        batch_size=2048,
        config_root=None
    )
    react_metrics = react_evaluator.eval_ood(fsood=False)
    print('ReAct metrics:')
    print(react_metrics)
    results_df = pd.DataFrame(react_metrics)
    results_df.to_csv('ood_results_react.csv', index=False)
    print("Results saved to ood_results_react.csv")


    print("Running SCALE evaluation...")
    scale_evaluator = Evaluator(
        net,
        id_name=ID_NAME,
        data_root=DATA_ROOT,
        postprocessor_name='scale',
        batch_size=2048,
        config_root=None
    )
    scale_metrics = scale_evaluator.eval_ood(fsood=False)
    print('SCALE metrics:')
    print(scale_metrics)
    results_df = pd.DataFrame(scale_metrics)
    results_df.to_csv('ood_results_scale.csv', index=False)
    print("Results saved to ood_results_scale.csv")

    # save all metrics to a csv file



    full_results = pd.concat(
        [
            custom_metrics,
            diffpath_metrics,
            diffnll_metrics,
            msma_metrics,
            ddpm_metrics,
            odin_metrics,
            ebo_metrics,
            react_metrics,
            scale_metrics
        ],
        keys=[
            'Ours',
            'DiffPath',
            'DiffNLL',
            'MSMA',
            'DDPM',
            'ODIN',
            'EBO',
            'ReAct',
            'SCALE'
        ],
        names=['Method', 'Dataset']
    ).reset_index(level=1).rename(columns={'level_1': 'Dataset'})

    full_results.to_csv('ood_results.csv')
    print(full_results)
    # Save the metrics to a CSV file

if __name__ == '__main__':
    main()

