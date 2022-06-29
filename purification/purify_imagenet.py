import torch
from utils import *
import tqdm
import pytorch_ssim

def purify_imagenet(x, diffusion,model,  max_iter, mode, config):
    # From noisy initialized image to purified image
    images_list = []
    transform_raw_to_diff = raw_to_diff(config.structure.dataset)
    transform_diff_to_raw = diff_to_raw(config.structure.dataset)
    x_adv = transform_raw_to_diff(x).to(config.device.diff_device)
    x_adv = torch.nn.functional.interpolate(x_adv, size=[256, 256], mode="bilinear")  # transfrom size 224 -> 256

    t_steps = torch.ones(x_adv.shape[0], device=config.device.diff_device).long()
    t_steps = t_steps * (config.purification.purify_step-1)
    shape = list(x_adv.shape)
    model_kwargs = {}

    def cond_fn(x_reverse_t, t):
        """
        Calculate the grad of guided condition.
        """
        print(f'cond_fn{t}')
        with torch.enable_grad():
            x_in = x_reverse_t.detach().requires_grad_(True)
            
            # x_adv_t = diffusion.q_sample(x_adv, t)
            x_adv_t = x_adv
            # scale = exp(config.purification.guide_exp_a * t / config.purification.purify_step+config.purification.guide_exp_b) + config.purification.guide_scale_base
            if config.purification.guide_mode == 'MSE': 
                selected = -1 * F.mse_loss(x_in, x_adv_t)
                scale = diffusion.compute_scale(x_in,t, config.attack.ptb*2/255. / 3. / config.purification.guide_scale)
            elif config.purification.guide_mode == 'SSIM':
                selected = pytorch_ssim.ssim(x_in, x_adv_t)
                scale = diffusion.compute_scale(x_in,t, config.attack.ptb*2/255. / 3. / config.purification.guide_scale)
            elif config.purification.guide_mode == 'CONSTANT': 
                selected = pytorch_ssim.ssim(x_in, x_adv_t)
                scale = config.purification.guide_scale
            return torch.autograd.grad(selected.sum(), x_in)[0] * scale


    with torch.no_grad():
        images = []
        xt_reverse = x_adv
        for i in range(max_iter):            
            adv_sample = diffusion.q_sample(xt_reverse,t_steps)
            sample_fn = diffusion.p_sample_loop if not config.net.use_ddim else diffusion.ddim_sample_loop
            xt_reverse = sample_fn(
                    model,
                    shape,
                    num_purifysteps = config.purification.purify_step,
                    noise = adv_sample,
                    clip_denoised=config.net.clip_denoised,
                    cond_fn = cond_fn if config.purification.cond else None,
                    model_kwargs=model_kwargs,
                )
            x_pur_t = xt_reverse.clone().detach()
            x_pur = torch.clamp(transform_diff_to_raw(x_pur_t), 0.0, 1.0)
            x_pur = torch.nn.functional.interpolate(x_pur, size=[224, 224], mode="bilinear") # transfrom size 256 -> 224
            images.append(x_pur)

    return images
