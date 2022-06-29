from .evasion import *
import torch
import numpy as np

def _create_random_target(label,device):
    """
    we consider targeted attacks when
    evaluating under the white-box settings, where the targeted
    class is selected uniformly at random
    """
    label_offset = torch.randint(1, 1000, label.shape).to(device)
    return torch.remainder(label + label_offset, 1000).to(device)

def targeted_pgd(x, y, diffusion, network_clf, config,model = None):
    x = x.to(config.device.clf_device)
    y = _create_random_target(y,config.device.clf_device).to(config.device.clf_device)
    print(y)
    if config.attack.ball_dim==-1:
        x_adv = projected_gradient_descent(network_clf, x, eps=config.attack.ptb/255., eps_iter=config.attack.ptb/255./4, nb_iter=config.attack.attack_steps, norm=np.inf,clip_min=0.,clip_max=1.,y=y,targeted=True).to(config.device.clf_device)
    return x_adv, None ,  None