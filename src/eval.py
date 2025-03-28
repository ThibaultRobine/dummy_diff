import torch
from openood.evaluation_api import Evaluator
from openood.networks import ResNet18_32x32
from diffusion_postprocessor import DiffusionPostprocessor  # Your custom class

# 1. Load pre-trained classifier (example: CIFAR-10 ResNet)
net = ResNet18_32x32(num_classes=10)
net.load_state_dict(torch.load('./pretrained_models/cifar10_res18_v1.5/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt'))
net.cuda().eval()

# 2. Initialize YOUR postprocessor (no YAML needed)
config = {
    'postprocessor_args': {
        'diffusion_steps': 100,  # Custom args for your method
    }
}
custom_postprocessor = DiffusionPostprocessor(config)

# 3. Evaluate with OpenOOD
evaluator = Evaluator(
    net,
    id_name='cifar10', 
    data_root='./data',
    postprocessor=custom_postprocessor,  # Pass instance directly
    batch_size=200,
)

metrics = evaluator.eval_ood(fsood=False)
print(metrics)