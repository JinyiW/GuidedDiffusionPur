from math import exp
import torch
from utils import *
import tqdm
import pytorch_ssim


def diff_purify(x, diffusion, max_iter, mode, config):
    # From noisy initialized image to purified image
    images_list = []
    transform_raw_to_diff = raw_to_diff(config.structure.dataset)
    transform_diff_to_raw = diff_to_raw(config.structure.dataset)
    x_adv = transform_raw_to_diff(x).to(config.device.diff_device)

    def cond_fn(x_reverse_t, t):
        """
        Calculate the grad of guided condition.
        """
        with torch.enable_grad():
            x_in = x_reverse_t.detach().requires_grad_(True)
            x_adv_t = diffusion.diffuse_t_steps(x_adv, t)
            # scale = exp(config.purification.guide_exp_a * t / config.purification.purify_step+config.purification.guide_exp_b) + config.purification.guide_scale_base
            if config.purification.guide_mode == 'MSE': 
                selected = -1 * F.mse_loss(x_in, x_adv_t)
                scale = diffusion.compute_scale(x_in,t, config.attack.ptb*2/255. / 3. / config.purification.guide_scale)
            elif config.purification.guide_mode == 'SSIM':
                selected = pytorch_ssim.ssim(x_in, x_adv_t)
                scale = diffusion.compute_scale(x_in,t, config.attack.ptb*2/255. / 3. / config.purification.guide_scale)
            elif config.purification.guide_mode == 'CONSTANT': 
                scale = config.purification.guide_scale
            return torch.autograd.grad(selected.sum(), x_in)[0] * scale

    with torch.no_grad():
        images = []
        xt_reverse = x_adv
        for i in range(max_iter):
            # # method 1: save every step pic
            # images = []
            # xt = diffusion.diffuse_t_steps(x, config.purification.purify_step)
            # for j in range(config.purification.purify_step):
            #     xt = diffusion.denoise(xt.shape[0], n_steps=1, x=xt.to(config.device.diff_device), curr_step=(config.purification.purify_step-j), progress_bar=tqdm.tqdm)
            #     x_pur_t = xt.clone().detach()
            #     x_pur = torch.clamp(transform_diff_to_raw(x_pur_t), 0.0, 1.0)
            #     images.append(x_pur)
            # images_list.append(images)
            
            # method 2: save final step pic
            
            xt = diffusion.diffuse_t_steps(xt_reverse, config.purification.purify_step)

            #save_images
            save_images(diffusion.diffuse_t_steps(xt_reverse, 400),'/mnt/lustre/wangjinyi/Diff-purify/generate_cifar10/diffuse400')
            save_images(diffusion.diffuse_t_steps(xt_reverse, 200),'/mnt/lustre/wangjinyi/Diff-purify/generate_cifar10/diffuse200')
            save_images(diffusion.diffuse_t_steps(xt_reverse, 100),'/mnt/lustre/wangjinyi/Diff-purify/generate_cifar10/diffuse100')
            save_images(diffusion.diffuse_t_steps(xt_reverse, 0),'/mnt/lustre/wangjinyi/Diff-purify/generate_cifar10/diffuse0')

            xt_reverse = diffusion.denoise(
                xt.shape[0], 
                n_steps=config.purification.purify_step, 
                x=xt.to(config.device.diff_device), 
                curr_step=config.purification.purify_step, 
                # progress_bar=tqdm.tqdm,
                cond_fn = cond_fn if config.purification.cond else None
            )

            #save_images
            x200 = diffusion.denoise(
                xt.shape[0], 
                n_steps=200, 
                x=xt.to(config.device.diff_device), 
                curr_step=400, 
                # progress_bar=tqdm.tqdm,
                cond_fn = cond_fn if config.purification.cond else None
            )
            x100 = diffusion.denoise(
                xt.shape[0], 
                n_steps=100, 
                x=x200.to(config.device.diff_device), 
                curr_step=200, 
                # progress_bar=tqdm.tqdm,
                cond_fn = cond_fn if config.purification.cond else None
            )
            x000 = diffusion.denoise(
                xt.shape[0], 
                n_steps=100, 
                x=x100.to(config.device.diff_device), 
                curr_step=100, 
                # progress_bar=tqdm.tqdm,
                cond_fn = cond_fn if config.purification.cond else None
            )
            save_images(x200,'/mnt/lustre/wangjinyi/Diff-purify/generate_cifar10/reverse200')
            save_images(x100,'/mnt/lustre/wangjinyi/Diff-purify/generate_cifar10/reverse100')
            save_images(x000,'/mnt/lustre/wangjinyi/Diff-purify/generate_cifar10/reverse0')
            save_images(xt_reverse,'/mnt/lustre/wangjinyi/Diff-purify/generate_cifar10/reverse')

            x_pur_t = xt_reverse.clone().detach()
            x_pur = torch.clamp(transform_diff_to_raw(x_pur_t), 0.0, 1.0)
            images.append(x_pur)

    return images
