from advertorch.attacks import LinfSPSAAttack

### Classifer PGD attack
# Attack input x
def SPSA(x, y, diffusion, network_clf, config):
    x = x.to(config.device.clf_device)
    y = y.to(config.device.clf_device)
    attack_kwargs = dict(
        eps=config.attack.ptb/255, delta=0.01, lr=0.01, nb_iter=100, nb_sample=128,
        max_batch_size=1280, targeted=False,
        loss_fn=None,
        clip_min=0.0, clip_max=1.0)
    fmodel = LinfSPSAAttack(network_clf,**attack_kwargs)
    x_adv = fmodel.perturb(x, y)
    return x_adv, None , None