import foolbox
import torch
from utils import *
from foolbox.criteria import TargetedMisclassification
### Classifer PGD attack
# Attack input x

def _create_random_target(label,device):
    """
    we consider targeted attacks when
    evaluating under the white-box settings, where the targeted
    class is selected uniformly at random
    """
    label_offset = torch.randint(1, 1000, label.shape).to(device)
    return torch.remainder(label + label_offset, 1000).to(device)

def clf_pgd(x, y, diffusion, network_clf, config,model = None):
    x = x.to(config.device.clf_device)
    y = y.to(config.device.clf_device)
    if config.attack.if_targeted:
        print(y)
        y = _create_random_target(y,config.device.clf_device)
        print(y)
        y = TargetedMisclassification(y)
    fmodel = foolbox.PyTorchModel(network_clf, device=config.device.clf_device, bounds=(0., 1.), preprocessing=foolbox_preprocess(config.structure.dataset))
    if config.attack.ball_dim==-1:
        attack = foolbox.attacks.LinfPGD(rel_stepsize=0.25,steps=config.attack.attack_steps) # Can be modified for better attack
        _, x_adv, success = attack(fmodel, x, y, epsilons=config.attack.ptb/256.)
        acc = 1 - success.float().mean(axis=-1)
    elif config.attack.ball_dim==2:
        attack = foolbox.attacks.L2PGD(rel_stepsize=0.25,steps=config.attack.attack_steps) # Can be modified for better attack
        _, x_adv, success = attack(fmodel, x, y, epsilons=config.attack.ptb/256.)
        acc = 1 - success.float().mean(axis=-1)
    return x_adv, success, acc