import torch

def preprocess(x, dset):
    if dset=="CIFAR10-C":
        return x / 255.
    else:
        return x

def foolbox_preprocess(dset):
    if dset in ["CIFAR10", "CIFAR10-C"]:
        return dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], axis=-3)
    elif dset in ["CIFAR100"]:
        return dict(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761], axis=-3)
    elif dset in ["ImageNet"]:
        return dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    else:
        return dict(mean=[0., 0., 0.], std=[1., 1., 1.], axis=-3)