from autoattack import AutoAttack

def autoattack(x, y, diffusion, network_clf, config,model = None):
    x = x.to(config.device.clf_device)
    y = y.to(config.device.clf_device)
    adversary = AutoAttack(network_clf, norm='Linf', eps=config.attack.ptb/255., version='standard', device=config.device.clf_device)
    x_adv = adversary.run_standard_evaluation(x, y, bs=config.structure.bsize).to(config.device.clf_device)
    return x_adv,None ,None