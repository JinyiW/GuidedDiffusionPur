from scipy.fftpack import diff
from utils import *
from clf_models.networks import *
from attacks import *
import pandas as pd
from purification.diff_purify import *
from pytorch_diffusion.diffusion import Diffusion

import tqdm
from PIL import Image

def normalize_and_quantize(xt):
    # we assume xt range from -1 to 1
    # we make it integer values from 0 to 255
    if xt.max() > 1 or xt.min() < -1:
        center = (xt.max() + xt.min()) / 2
        xt = xt - center
        xt = xt / (xt.max() - xt.min())
    xt = xt.transpose(1,2).transpose(2,3) # 1 H W C
    xt = xt[0].cpu().detach().numpy() # H W C

    xt = (xt+1) / 2 * 255
    xt = xt.astype(np.uint8)
    return xt

def save_single_xt_reverse(xt_reverse, diff_step, reverse_step, index, save_dir):
    # 1 C H W
    xt_reverse = normalize_and_quantize(xt_reverse)
    
    save_name = '%d_diff_%d_rev_%d.png' % (index, diff_step, reverse_step) 
    Image.fromarray(xt_reverse).save(os.path.join(save_dir, save_name))

def save_xt_reverse(xt_reverse, diff_step, reverse_step, start_index, save_dir, y):
    # xt_reverse = xt_reverse.detach().cpu().numpy()
    for i in range(xt_reverse.shape[0]):
        index = start_index + i
        xt_reverse_i = xt_reverse[i:i+1]
        label_i = y[i]
        save_root = os.path.join(save_dir, f'{label_i}')
        os.makedirs(save_root, exist_ok=True)
        save_single_xt_reverse(xt_reverse_i, diff_step, reverse_step, index, save_root)

def diff_reverse_gen(x, diffusion, diff_step, reverse_step, sample_number, start_index, save_dir, y):
    for j in range(sample_number):
        xt = diffusion.diffuse_t_steps(x, diff_step)
        xt_reverse = diffusion.denoise(xt.shape[0], n_steps=reverse_step, x=xt.to("cuda:0"), curr_step=diff_step, progress_bar=tqdm.tqdm)
        save_xt_reverse(xt_reverse, diff_step, reverse_step, start_index+j*xt_reverse.shape[0], save_dir, y)

if __name__ == "__main__":

    diff_step = 60
    reverse_step = 50
    sample_number = 1
    batch_size = 100
    
    # import data (tensor)
    testLoader = importData(dataset="CIFAR10", train=True, shuffle=False, bsize=batch_size)

    #import Diff Pretrained network
    model_name = 'ema_cifar10'
    diffusion = Diffusion.from_pretrained(model_name, device="cuda:0")

    #Diff
    save_dir = os.path.join("/home/wangjinyi/generated_samples",'diff_%d_rev_%d' % (diff_step, reverse_step) )
    transform_raw_to_diff = raw_to_diff("CIFAR10")
    for i, (x,y) in enumerate(testLoader):
        print('Progress [%d/%d]' % (i, len(testLoader)), flush=True)
        x = transform_raw_to_diff(x).to("cuda:0")
        diff_reverse_gen(x, diffusion, diff_step, reverse_step, sample_number, i*batch_size*sample_number, save_dir, y)


