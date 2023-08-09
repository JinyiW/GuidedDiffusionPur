# Guided Diffusion Model for Adversarial Purification
### by [Jinyi Wang], [Zhaoyang Lyu], [Bo Dai], [Hongfei Fu]

This repository includes the official PyTorch implementation of our [paper](https://arxiv.org/abs/2205.14969):

```
@InProceedings{pmlr-v162-nie22a,
  title = 	 {Diffusion Models for Adversarial Purification},
  author =       {Nie, Weili and Guo, Brandon and Huang, Yujia and Xiao, Chaowei and Vahdat, Arash and Anandkumar, Animashree},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {16805--16827},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/nie22a/nie22a.pdf},
}
```


## What does our work do?
We propose a method that gives adversarial robustness to a neural network model against (stochastic) adversarial attacks by using an Guided Diffusion Model.

## Running Codes
### Dependency
Run the following command to install some necessary Python 3 packages by anaconda to run our code.
```
conda env create -f environment.yml
```

### Running code
To run the experiments, enter the following command.
```
python main.py --config <config-file>
```
For example, we provide the example configuration file `configs/ImageNet_PGD.yml` in the repository.

### Parallel Running code
To run the experiments parallelly, enter the following command.
```
python parallel_run.py --device 8 --rank 0 --world_size 8 --config ImageNet_Res50.yml
```
For example, we provide the example configuration file `configs/cifar10_bpda_eot_sigma025_eot15.yml` in the repository.

### Attack Methods
For adversarial attacks, the classifier PGD attack and BPDA+EOT attack are implemented in `attacks/clf_pgd.py` and `attacks/bpda_strong.py`, respectively. At the configuration file, setting the `attack.attack_method` into `clf_pgd` or `bpda_strong` will run these attacks, respectively.


### Main components
| File name | Explanation | 
|:-|:-|
| `main.py` | Execute the main code, with initializing configurations and loggers. |
| `runners/empirical.py` | Attacks and purifies the image to show empirical adversarial robustness. |
| `attacks/bpda_strong.py` | Code for BPDA+EOT attack. |
| `purification/adp.py` | Code for adversarial purification. |
| `guided_diffusion/*` | Code for DDPM on ImageNet. |
| `pytorch_diffusion/*` | Code for DDPM on CIFAR-10. |
| `networks/*` | Code for used classifier network architectures. |
| `utils/*` | Utility files. |

### Notes
* For the configuration files, we use the pixel ranges `[0, 255]` for the perturbation scale `attack.ptb` and the one-step attack scale `attack.alpha`. And the main experiments are performed within the pixel range `[0, 1]` after being rescaled during execution.
* For training the EBM and classifier models, we primarily used the pre-existing methods such as [256*256_pretrained_diffusion_model](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) and WideResNet classifier. [Here](https://github.com/meliketoy/wide-resnet.pytorch) is the repository we used for training the WideResNet classifier. 


## Contact
For further details, please contact `jinyi.wang@sjtu.edu.cn`.

## License
MIT

This implementation is based on / inspired by:

- [https://github.com/openai/guided-diffusion](https://github.com/openai/guided-diffusion) (Pytorch DDPM on ImageNet)
- [https://github.com/pesser/pytorch_diffusion](https://github.com/pesser/pytorch_diffusion) (PyTorch helper that loads the DDPM model), and
- [https://github.com/jmyoon1/adp](https://github.com/jmyoon1/adp) (code structure and attack algorithms).
