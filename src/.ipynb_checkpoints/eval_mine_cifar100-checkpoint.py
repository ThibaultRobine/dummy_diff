# eval_mine.py
import torch
import pandas as pd
from openood.evaluation_api import Evaluator
from openood.networks import ResNet18_32x32
from diffusion_postprocessor_mine import CustomDiffusionPostprocessor
from ddpm_ood_postprocessor_mine import DdpmOODPostprocessor
from diffpath_postprocessor_mine import DiffPathPostprocessor
from msma_postprocessor_mine import MSMAPostprocessor
from diffusion_nll_postprocessor_mine import DiffusionNllPostprocessor
from diffusion_model_manager_mine import set_trained_diffusion_model
from model_unet import DiscreteDDPMProcess, UniformDiscreteTimeSampler, DiffusionModel
import dataclasses
import timm

ID_NAME = 'cifar100'
DATA_ROOT = './data'

def main():
    # Load pretrained model
    net = ResNet18_32x32(num_classes=100)
    net.load_state_dict(torch.load(
        'pretrained_models/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt'
    ))
    net.cuda().eval()

    # Configuration with all hyperparameters
    config = {
        'postprocessor': {
            'name': 'custom_diffusion_postprocessor.CustomDiffusionPostprocessor',
            'APS_mode': False,
            'diffusion_args': {
                # Diffusion process parameters
                'num_diffusion_timesteps': 5000,
                'schedule_type': 'linear',
                
                
                # Training parameters
                'train_loop': {
                    'batch_size': 4096*4,
                    'eta_min' : 3e-12,
                    'lr': 3e-4,
                    'warmup_steps': 1000,
                    'total_steps': 25000,  # Reduced for testing
                    'max_step': 1000+25000,
                },
                    'unet': {
        'activation': 'elu',
        'time_embedding_dim': 1024*5
    }
            },
            'integration_method': {
                # Reference point optimization
                'kmeans_k': 40,
                'asc_steps': 10000,
                'asc_lr': 1e-2,
                'asc_grad_clip': 10.0,
                'asc_wd': 0.0,
                
                # Score integration
                'gauss_n': 25,
                'gauss_batch': 5000,
                
                # Convergence criteria
                'asc_convergence_window': 1000,
                'asc_convergence_threshold': 5e-1,
                'asc_min_steps': 100,
                
                # Device settings
                'device': 'cuda'
            },
            'ddpmood_args': {
            'ood_t' : 250,
            'num_inference_steps' : 100,
        },
            'diffpath_args': {
            'n_steps': 20,#20,
            'batch_size': 2048,
            'device': 'cuda',
        },
            'msma_args': {
                'n_steps': 20,
                'batch_size': 2048,
                'device': 'cuda',
            },
            'diffusion_nll': {
                'batch_size': 2048*2,
                'device': 'cuda',
            },
    }
}
    #Load the diffusion model
    diffusion_params = config['postprocessor']['diffusion_args']
    net_config = config['postprocessor']['integration_method']
    checkpoint = torch.load('model_checkpoint_latent_linear_100.pth')
    diffusion_process = DiscreteDDPMProcess(num_diffusion_timesteps=diffusion_params['num_diffusion_timesteps'],schedule_type=diffusion_params['schedule_type'])
    time_sampler = UniformDiscreteTimeSampler(diffusion_process.tmin, diffusion_process.tmax)
    unet_params = diffusion_params['unet']
    @dataclasses.dataclass
    class NetConfig:
        activation: str = unet_params.get('activation')
        time_embedding_dim: int = unet_params.get('time_embedding_dim')

    diffusion_model = DiffusionModel(
        diffusion_process=diffusion_process,
        time_sampler=time_sampler,
        net_config=NetConfig(),
        data_shape=(512,)
    ).cuda()
    diffusion_model.load_state_dict(checkpoint['model_state_dict'])
    diffusion_model.eval()
    diffusion_model.cuda()
    set_trained_diffusion_model(diffusion_model)


    # diffnll_postprocessor = DiffusionNllPostprocessor(config)
    # print("Running DIFFNLL evaluation...")
    # diffnll_evaluator = Evaluator(
    #     net,
    #     id_name=ID_NAME,
    #     data_root=DATA_ROOT,
    #     postprocessor=diffnll_postprocessor,
    #     batch_size=2048,
    #     config_root=None
    # )
    # diffnll_metrics = diffnll_evaluator.eval_ood(fsood=False)
    # print('DIFFNLL metrics:')
    # print(diffnll_metrics)
    # results_df = pd.DataFrame(diffnll_metrics)
    # results_df.to_csv('ood_results_diffnll_100.csv', index=False)
    # print("Results saved to ood_results_diffnll_100.csv")

    # msma_postprocessor = MSMAPostprocessor(config)
    # print("Running MSMA evaluation...")
    # msma_evaluator = Evaluator(
    #     net,
    #     id_name=ID_NAME,
    #     data_root=DATA_ROOT,
    #     postprocessor=msma_postprocessor,
    #     batch_size=2048,
    #     config_root=None
    # )
    # msma_metrics = msma_evaluator.eval_ood(fsood=False)
    # print('MSMA metrics:')
    # print(msma_metrics)
    # results_df = pd.DataFrame(msma_metrics)
    # results_df.to_csv('ood_results_msma_100.csv', index=False)
    # print("Results saved to ood_results_msma_100.csv")

    # diffpath_postprocessor = DiffPathPostprocessor(config)
    # print("Running DIFFPATH evaluation...")
    # diffpath_evaluator = Evaluator(
    #     net,
    #     id_name=ID_NAME,
    #     data_root=DATA_ROOT,
    #     postprocessor=diffpath_postprocessor,
    #     batch_size=2048,
    #     config_root=None
    # )
    # diffpath_metrics = diffpath_evaluator.eval_ood(fsood=False)
    # print('DIFFPATH metrics:')
    # print(diffpath_metrics)
    # results_df = pd.DataFrame(diffpath_metrics)
    # results_df.to_csv('ood_results_diffpath_100.csv', index=False)
    # print("Results saved to ood_results_diffpath_100.csv")

    # ddpm_ood_postprocessor = DdpmOODPostprocessor(config)
    # print("Running DDPM evaluation...")
    # ddpm_evaluator = Evaluator(
    #     net,
    #     id_name=ID_NAME,
    #     data_root=DATA_ROOT,
    #     postprocessor=ddpm_ood_postprocessor,
    #     batch_size=2048,
    #     config_root=None
    # )
    # ddpm_metrics = ddpm_evaluator.eval_ood(fsood=False)
    # print('DDPM metrics:')
    # print(ddpm_metrics)
    # results_df = pd.DataFrame(ddpm_metrics)
    # results_df.to_csv('ood_results_ddpm_100.csv', index=False)
    # print("Results saved to ood_results_ddpm_100.csv")

    
    # Initialize and run evaluation
    print("Initializing custom diffusion postprocessor...")
    custom_postprocessor = CustomDiffusionPostprocessor(config)
    
    print("Running evaluation...")
    evaluator = Evaluator(
        net,
        id_name=ID_NAME,
        data_root=DATA_ROOT,
        postprocessor=custom_postprocessor,
        batch_size=2048,
        config_root=None
    )
    
    metrics = evaluator.eval_ood(fsood=False)
    
    # Save results
    print('\nFinal metrics:')
    print(metrics)
    pd.DataFrame(metrics).to_csv('ood_results_test_100.csv', index=False)
    print("Results saved to ood_results_test_100.csv")


    # print("Running ODIN evaluation...")
    # odin_evaluator = Evaluator(
    #     net,
    #     id_name=ID_NAME,
    #     data_root=DATA_ROOT,
    #     postprocessor_name='odin',
    #     batch_size=2048,
    #     config_root=None
    # )
    # odin_metrics = odin_evaluator.eval_ood(fsood=False)
    # print('ODIN metrics:')
    # print(odin_metrics)
    # results_df = pd.DataFrame(odin_metrics)
    # results_df.to_csv('ood_results_odin_100.csv', index=False)
    # print("Results saved to ood_results_odin_100.csv")


    # print("Running EBO evaluation...")
    # ebo_evaluator = Evaluator(
    #     net,
    #     id_name=ID_NAME,
    #     data_root=DATA_ROOT,
    #     postprocessor_name='ebo',
    #     batch_size=2048,
    #     config_root=None
    # )
    # ebo_metrics = ebo_evaluator.eval_ood(fsood=False)
    # print('EBO metrics:')
    # print(ebo_metrics)
    # results_df = pd.DataFrame(ebo_metrics)
    # results_df.to_csv('ood_results_ebo_100.csv', index=False)
    # print("Results saved to ood_results_ebo_100.csv")

    # print("Running ReAct evaluation...")
    # react_evaluator = Evaluator(
    #     net,
    #     id_name=ID_NAME,
    #     data_root=DATA_ROOT,
    #     postprocessor_name='react',
    #     batch_size=2048,
    #     config_root=None
    # )
    # react_metrics = react_evaluator.eval_ood(fsood=False)
    # print('ReAct metrics:')
    # print(react_metrics)
    # results_df = pd.DataFrame(react_metrics)
    # results_df.to_csv('ood_results_react_100.csv', index=False)
    # print("Results saved to ood_results_react_100.csv")


    # print("Running SCALE evaluation...")
    # scale_evaluator = Evaluator(
    #     net,
    #     id_name=ID_NAME,
    #     data_root=DATA_ROOT,
    #     postprocessor_name='scale',
    #     batch_size=2048,
    #     config_root=None
    # )
    # scale_metrics = scale_evaluator.eval_ood(fsood=False)
    # print('SCALE metrics:')
    # print(scale_metrics)
    # results_df = pd.DataFrame(scale_metrics)
    # results_df.to_csv('ood_results_scale_100.csv', index=False)
    # print("Results saved to ood_results_scale_100.csv")

    # # # # # save all metrics to a csv file



    # full_results = pd.concat(
    #     [
    #         pd.DataFrame(diffnll_metrics),
    #         pd.DataFrame(msma_metrics),
    #         pd.DataFrame(diffpath_metrics),
    #         pd.DataFrame(ddpm_metrics),
    #         pd.DataFrame(metrics),
    #         pd.DataFrame(odin_metrics),
    #         pd.DataFrame(ebo_metrics),
    #         pd.DataFrame(react_metrics),
    #         pd.DataFrame(scale_metrics)
    #     ],
    #     keys=[
    #         'diffnll',
    #         'msma',
    #         'diffpath',
    #         'ddpm',
    #         'custom_diffusion',
    #         'odin',
    #         'ebo',
    #         'react',
    #         'scale'
    #     ],
    #     names=['method', 'Dataset']
    # ).reset_index(level=1).rename(columns={'level_1': 'Dataset'})

    # full_results.to_csv('ood_results_full_100.csv')
    # print("All results saved to ood_results_full_100.csv")
    # print(full_results)




if __name__ == '__main__':
    main()