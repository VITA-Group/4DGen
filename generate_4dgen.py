import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from skimage.io import imsave

from ldm.models.diffusion.sync_dreamer import SyncMultiviewDiffusion, SyncDDIMSampler
from ldm.util import instantiate_from_config, prepare_inputs


def load_model(cfg,ckpt,strict=True):
    config = OmegaConf.load(cfg)
    model = instantiate_from_config(config.model)
    print(f'loading model from {ckpt} ...')
    ckpt = torch.load(ckpt,map_location='cpu')
    model.load_state_dict(ckpt['state_dict'],strict=strict)
    model = model.cuda().eval()
    return model

import fire
def main(flags):
    torch.random.manual_seed(flags.seed)
    np.random.seed(flags.seed)

    model = load_model(flags.cfg, flags.ckpt, strict=True)
    assert isinstance(model, SyncMultiviewDiffusion)
    # Path(f'{flags.output}').mkdir(exist_ok=True, parents=True)

    # prepare data
    data, image_input_save = prepare_inputs(flags.input, flags.elevation, flags.crop_size)
    output_fn = f'{flags.output}.png'
    # output_fn = f'{flags.output.split("/")[-1]}.png'.replace('_sync/', '_pose0/')
    print('output_fn:',output_fn)
    image_input_save.save(output_fn)
    for k, v in data.items():
        data[k] = v.unsqueeze(0).cuda()
        data[k] = torch.repeat_interleave(data[k], flags.sample_num, dim=0)

    if flags.sampler=='ddim':
        sampler = SyncDDIMSampler(model, flags.sample_steps)
    else:
        raise NotImplementedError
    x_sample = model.sample(sampler, data, flags.cfg_scale, flags.batch_view_num)

    B, N, _, H, W = x_sample.shape
    x_sample = (torch.clamp(x_sample,max=1.0,min=-1.0) + 1) * 0.5
    x_sample = x_sample.permute(0,1,3,4,2).cpu().numpy() * 255
    x_sample = x_sample.astype(np.uint8)

    for bi in range(B):
        # output_fn = Path(flags.output)/ f'{bi}.png'
        output_fn = f'{flags.output}_{bi}.png'
        imsave(output_fn, np.concatenate([x_sample[bi,ni] for ni in range(N)], 1))


# def rr(inp, oup, xx):
    
import os
if __name__=="__main__":
    # # main()
    # for i in range(101, 126):
    #     print('Starts running', i)
    #     main(f"/home/dejia.xu/repo/Practical-RIFE/rose_rife_96fps/{i}_rgba.png", f"output_rife/rose{i}")
    # fire.Fire(rr)

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',type=str, default='configs/syncdreamer.yaml')
    parser.add_argument('--ckpt',type=str, default='ckpt/syncdreamer-pretrain.ckpt')
    parser.add_argument('--output', type=str, required=False)
    parser.add_argument('--input', type=str, required=False)
    parser.add_argument('--oup', type=str, required=False)
    parser.add_argument('--inp', type=str, required=False)
    parser.add_argument('--xx', type=int, default=0,required=False)
    parser.add_argument('--elevation', type=float, default=0)

    parser.add_argument('--sample_num', type=int, default=1)
    parser.add_argument('--crop_size', type=int, default=-1)
    parser.add_argument('--cfg_scale', type=float, default=2.0)
    parser.add_argument('--batch_view_num', type=int, default=8)
    parser.add_argument('--seed', type=int, default=6033)

    parser.add_argument('--sampler', type=str, default='ddim')
    parser.add_argument('--sample_steps', type=int, default=50)
    flags = parser.parse_args()
    for idx in range(flags.xx, flags.xx + 16):
        try:
            flags.input = flags.inp + f"/{idx}.png"
            # flags.input = flags.inp + f"/{idx}_rgba.png"
            flags.output = flags.oup + f"/{idx}"
            os.makedirs(flags.oup, exist_ok=True)
            #os.makedirs(flags.oup.replace('_sync/', '_pose0/'), exist_ok=True)
            # if not os.path.exists(flags.output):
            #     os.makedirs(os.path.basename(flags.output), exist_ok=True)
            main(flags)
        except Exception as e:
            print(e)
            # for i in range(32):
    #     main(f"/home/dejia.xu/repo/threestudio/in-the-wild/blooming_rose/{i}.png", f"output2/rose{i}")

