import torch
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, CIFAR100, ImageNet, ImageFolder 
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from utils.tinyimagenet import TINYIMAGENET
# from runners import TinyImageNet

import os
import sys
# path_root = '/home/wangjinyi'
# path_root = '/home/cpr'
path_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(path_root)

def importData(dataset, train, shuffle, bsize,distortion_name=None,severity=None):
    '''
    dataset: datasets (MNIST, CIFAR10, CIFAR100, SVHN, CELEBA)
    train: True if training set, False if test set
    shuffle: Whether to shuffle or not
    bsize: minibatch size
    '''
    # Set transform
    dataset_list = ["MNIST", "CIFAR10", "FashionMNIST", "CIFAR10-C", "CIFAR100", "ImageNet","ImageNet-C"]
    if dataset not in dataset_list:
        sys.exit("Non-handled dataset")

    if dataset=="MNIST":
        path = os.path.join(path_root, "datasets", "MNIST")
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = MNIST(path, train=train, download=True, transform=transform)
    elif dataset=="CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        path = os.path.join(path_root, "datasets", "CIFAR10")
        dataset = CIFAR10(path, train=train, download=True, transform=transform)
    elif dataset=="ImageNet":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()        
        ])
        path = os.path.join(path_root, "datasets", "ImageNet")
        dataset = ImageNet(path, split='val', transform=transform)
    elif dataset=="FashionMNIST":
        path = os.path.join(path_root, "datasets", "FashionMNIST")
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = FashionMNIST(path, train=train, download=True, transform=transform)
    elif dataset=="CIFAR100":
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        path = os.path.join(path_root, "datasets", "CIFAR100")
        dataset = CIFAR100(path, train=train, download=True, transform=transform)
    elif dataset=="TinyImageNet":
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        if train:
            dataset = TINYIMAGENET(os.path.join(path_root, 'datasets'), train=True)
        else:
            dataset = TINYIMAGENET(os.path.join(path_root, 'datasets'),train=False)
    elif dataset=="CIFAR10-C":
        dataloader = []
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        path = os.path.join(path_root, "datasets", "CIFAR-10-C")
        file_list = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', \
            'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
        label_path = os.path.join(path, "labels.npy")
        lb_file = np.load(label_path) # Size [50000]
        np_y = lb_file[0:10000]
        for i in range(len(file_list)):
            sub_dataloader = []
            np_x = np.load(os.path.join(path, file_list[i]+".npy"))
            np_x = np.transpose(np_x, (0,3,1,2))
            for j in range(5):
                tensor_x = torch.Tensor(np_x[j*10000:(j+1)*10000])
                tensor_y = torch.Tensor(np_y)
                dset = TensorDataset(tensor_x, tensor_y)
                sub_dataloader.append(DataLoader(dset, batch_size=bsize, shuffle=shuffle, num_workers=4))
            dataloader.append(sub_dataloader)
    elif dataset=="ImageNet-C":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        path = os.path.join(path_root, "datasets", "ImageNet-C")
        # file_list = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', \
        #     'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

        valdir = path +'/' + distortion_name + '/' + str(severity)
        dataloader = torch.utils.data.DataLoader(
            ImageFolder(valdir, transform),
            batch_size=bsize, shuffle=shuffle, num_workers=4, pin_memory=True)

        return dataloader
        

    if dataset != "CIFAR10-C":
        dataloader = DataLoader(dataset, batch_size=bsize, shuffle=shuffle, num_workers=4)
        return dataloader
    else:
        return dataloader
