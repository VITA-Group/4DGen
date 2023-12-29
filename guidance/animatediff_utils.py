from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
    StableDiffusionPipeline,
)
import torchvision.transforms.functional as TF

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('./')


import os
from omegaconf import OmegaConf
from einops import rearrange
import sys
import argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import io
from tqdm import tqdm
from datetime import datetime
import random
import imageio
from pathlib import Path
import shutil
import logging
from diffusers.utils.import_utils import is_xformers_available
# from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()  # disable warning
from animatediff.pipelines.pipeline_old import AnimationPipeline
# from animatediff.pipelines.pipeline_animation import AnimationPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import DDIMScheduler
from animatediff.models.unet import UNet3DConditionModel
from animatediff.utils.util import load_weights
from animatediff.utils.util import save_videos_grid

class AnimateDiff(nn.Module):
    def __init__(self, device='cuda',use_textual_inversion=False):
        inference_config=OmegaConf.load("animatediff/configs/inference/inference-v2.yaml")
        pretrained_model_path="animatediff/animatediff_models/StableDiffusion/stable-diffusion-v1-5"
        # pretrained_model_path="runwayml/stable-diffusion-v1-5"
        self.pretrained_model_path = pretrained_model_path
        tokenizer    = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
        if use_textual_inversion:
            inversion_path = None # TODO: CHANGE this!
            text_encoder = CLIPTextModel.from_pretrained(inversion_path, subfolder="checkpoint-500")
        else:
            text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
        vae          = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")            
        unet         = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.requires_grad_(False)
        if is_xformers_available(): unet.enable_xformers_memory_efficient_attention()
        else: assert False
        self.device = device = torch.device(device)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pipeline = AnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
        ).to(device)
        motion_module_path  = "./animatediff/animatediff_models/Motion_Module/mm_sd_v15_v2.ckpt"
        dreambooth_model_path      = "./animatediff/animatediff_models/DreamBooth_LoRA/rcnzCartoon3d_v20.safetensors"

        self.pipeline = load_weights(
            pipeline,
            motion_module_path         = motion_module_path,
            dreambooth_model_path      = dreambooth_model_path,
        ).to(device)
        # unet = unet.to(device)
        # vae = vae.to(device)
        # text_encoder = text_encoder.to(device)
        self.scheduler = self.pipeline.scheduler
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)
        # self.scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler", torch_dtype= torch.float32)
        self.rgb_to_latent = torch.from_numpy(np.array([[ 1.69810224, -0.28270747, -2.55163474, -0.78083445],
        [-0.02986101,  4.91430525,  2.23158593,  3.02981481],
        [-0.05746497, -3.04784101,  0.0448761 , -3.22913725]])).float().cuda(non_blocking=True) # 3 x 4
        self.latent_to_rgb = torch.from_numpy(np.array([
            [ 0.298,  0.207,  0.208],  # L1
            [ 0.187,  0.286,  0.173],  # L2
            [-0.158,  0.189,  0.264],  # L3
            [-0.184, -0.271, -0.473],  # L4
            ])).float().cuda(non_blocking=True) # 4 x 3

    def load_text_encoder(self, use_textual_inversion=False):
        if use_textual_inversion:
            inversion_path="."
            text_encoder = CLIPTextModel.from_pretrained(inversion_path, subfolder="checkpoint-500")
        else:
            text_encoder = CLIPTextModel.from_pretrained(self.pretrained_model_path, subfolder="text_encoder")
        return text_encoder

    @torch.no_grad()
    def prepare_text_emb(self, prompt=None, neg_prompt=None):
        #example
        if prompt is None:
            prompt = "a panda dancing"
            # prompt = "a <cvpr-panda> dancing"
        if neg_prompt is None:
            neg_prompt = "color distortion,color shift,green light,semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, immutable, unchanging, stable, fixed, permant, unvarying, stationary, constant, steady, motionless, inactive, still, rooted, set"
            # neg_prompt = "color distortion,color shift,green light,semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid,  mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions,  missing arms, missing legs, extra arms, extra legs"
        text_embeddings = self.pipeline._encode_prompt(
                [prompt], self.device, num_videos_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=[neg_prompt],
            )
        return text_embeddings

    @torch.no_grad()
    def prepare_text_emb_inversion(self, prompt=None, neg_prompt=None, inversion_prompt=None):
        #example
        if inversion_prompt is None:
            inversion_prompt = 'a <cvpr-panda> dancing'
        if prompt is None:
            prompt = "a panda dancing"
        if neg_prompt is None:
            neg_prompt = "color distortion,color shift,green light,semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid,  mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions,  missing arms, missing legs, extra arms, extra legs"
        text_embeddings = self.pipeline._encode_prompt(
                [prompt], self.device, num_videos_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=[neg_prompt],
            )
        # self.pipeline.text_encoder = self.load_text_encoder(use_textual_inversion=True)
        text_embeddings_inversion = self.pipeline._encode_prompt(
                [inversion_prompt], self.device, num_videos_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=[neg_prompt],
            )
        return text_embeddings, text_embeddings_inversion

    def get_cfg(self, noisy_latents, text_embeddings, guidance_scale, t):
        latent_model_input = torch.cat([noisy_latents] * 2)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        # print('latent_model_input', latent_model_input.shape)
        noise_pred = self.pipeline.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        return noise_pred
    
    def train_step(self, pred_rgb, text_embeddings, guidance_scale=30, as_latent=False):
        # shape = [1, 4, 8, 64, 64]   #b,c,t,h,w    b=1, c=4 beacause vae encode
        # latents_vsd = torch.randn(shape).to(device)  #input
        if not as_latent:
            print('diff input rgb', pred_rgb.shape)
            frame_size=pred_rgb.shape[0]
            # latents = (pred_rgb.permute(0, 2, 3, 1) @ self.rgb_to_latent).permute(3, 0, 1, 2)
            
            # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False).unsqueeze(0)
            # print('latents', latents.shape)
            latents = self.pipeline.vae.encode(pred_rgb * 2 - 1).latent_dist.sample() # [8, 4, 64, 64]
            print('latents shape',latents.shape)
            # randn_noise=torch.rand_like(latents[0]).to(latents.device)
            # for i in range(1,frame_size):
            #     i=torch.tensor(i,device=self.device).long()
            #     latents[i]=self.scheduler.add_noise(latents[i], randn_noise, i*100)
            latents = latents.unsqueeze(0).permute(0, 2, 1, 3, 4) * 0.18215   #[1, 4, 8, 32, 32])
            
            #image+guassian+guassian...

        else:
            latents = pred_rgb
        # latents = rearrange(latents, "b c) f h w -> (b f) c h w") 
        
        print('latents', latents.shape, latents.requires_grad)
        with torch.no_grad():
            noise = torch.randn_like(latents)
            # t=torch.tensor(100).to(device)   #
            t = torch.randint(
                    50, 950, (latents.shape[0],), device=self.device
                ).long()
            # print('time shape', t.shape)
            noisy_latents = self.scheduler.add_noise(latents, noise, t)
            noise_pred = self.get_cfg(noisy_latents, text_embeddings, guidance_scale, t)
            noise_diff = noise_pred - noise
        # noise_pred=self.pipeline(noisy_lantents=noisy_latents,
        #     t=t,
        #     prompt=prompt,
        #     negative_prompt= n_prompt,
        #     )
        # print('noise pred shape:',noise_pred.shape)  #([1, 4, 8, 64, 64])
        w = (1 - self.alphas[t]).view(noise.shape[0], 1, 1, 1, 1)
        grad = w * (noise_diff)
        grad = torch.nan_to_num(grad)

        # if not as_latent:
        #     # grad: [1, 4, 16, 64, 64]
        #     print(grad.shape)
        #     # norm = torch.norm(grad, dim=(1))
        #     norm = torch.norm(grad, dim=(1, 2))
        #     print(norm)
        #     thres = torch.ones_like(norm).detach() * 1
        #     # grad = torch.minimum(norm, thres) * F.normalize(grad, dim=(1))
        #     grad = torch.minimum(norm, thres) * F.normalize(grad, dim=(1, 2))

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum')
        return loss
    
    def train_step_inversion(self, pred_rgb, text_embeddings, text_embeddings_inversion, guidance_scale=30, as_latent=False):
        # shape = [1, 4, 8, 64, 64]   #b,c,t,h,w    b=1, c=4 beacause vae encode
        # latents_vsd = torch.randn(shape).to(device)  #input
        if not as_latent:
            print('diff input rgb', pred_rgb.shape)
            # latents = (pred_rgb.permute(0, 2, 3, 1) @ self.rgb_to_latent).permute(3, 0, 1, 2)
            
            # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False).unsqueeze(0)
            # print('latents', latents.shape)
            latents = self.pipeline.vae.encode(pred_rgb * 2 - 1).latent_dist.sample() # [8, 4, 64, 64]
            latents = latents.unsqueeze(0).permute(0, 2, 1, 3, 4) * 0.18215
        else:
            latents = pred_rgb
        # latents = rearrange(latents, "b c) f h w -> (b f) c h w") 
        
        print('latents', latents.shape, latents.requires_grad)
        with torch.no_grad():
            noise = torch.randn_like(latents)
            # t=torch.tensor(100).to(device)   #
            t = torch.randint(
                    50, 950, (latents.shape[0],), device=self.device
                ).long()
            # print('time shape', t.shape)
            noisy_latents = self.scheduler.add_noise(latents, noise, t)
            noise_pred_original = self.get_cfg(noisy_latents, text_embeddings, guidance_scale, t)
            noise_pred_inversion = self.get_cfg(noisy_latents, text_embeddings_inversion, guidance_scale, t)
            noise_diff = noise_pred_inversion - noise_pred_original
        print('noise pred shape:',noise_diff.shape)  #([1, 4, 8, 64, 64])
        w = (1 - self.alphas[t]).view(noise.shape[0], 1, 1, 1, 1)
        grad = w * (noise_diff)
        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum')
        return loss
    
    @torch.no_grad()
    def sample(self, text_embeddings, guidance_scale=7.5):
        # latents = self.pipeline.vae.encode(pred_rgb).latent_dist.mode()
        latents = torch.randn([1, 4, 16, 64, 64], device=self.device) * self.scheduler.init_noise_sigma
        # noise = torch.randn_like(latents)
        # t=torch.tensor(100).to(device)   #
        # t = torch.randint(
        #         0, self.diffusion_model.num_timesteps, (pred_rgb.shape[0],), device=self.device
        #     ).long()
        # print('time shape', t.shape)
        from tqdm import tqdm
        extra_step_kwargs = {}
        # if accepts_eta:
        extra_step_kwargs["eta"] = 0.0
        for i, t in enumerate(tqdm(self.scheduler.timesteps)):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self.pipeline.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, eta=0.0).prev_sample
        latents = 1 / 0.18215 * latents
        print('output', latents.shape)
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        imgs = self.pipeline.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def decode_latent(self, x):
        latents = 1 / 0.18215 * x
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        return (self.pipeline.vae.decode(latents).sample / 2 + 0.5).clamp(0, 1)

if __name__ == '__main__':
    torch.manual_seed(16931037867122267877)
    anim = AnimateDiff()
    text_emb = anim.prepare_text_emb()
    t2i = False
    if t2i:
        anim.scheduler.set_timesteps(50)
        # pred_rgb = torch.randn((8, 3, 256, 256)).cuda()
        # pred_rgb = torch.randn((1, 4, 8, 64, 64))
        res = anim.sample(text_emb)
        print(res.shape)
        res = res.permute(0, 2, 3, 1)
        res = res.detach().cpu().numpy()
        res = (res * 255).astype(np.uint8)
        print(res.shape)
        imageio.mimwrite('a.mp4', res, fps=16, quality=8, macro_block_size=1)
    sds = True
    if sds:
        prefix = 'inversion_sds_latent_0.01'
        # prefix = 'sds_rgb'
        from PIL import Image
        from torchvision.transforms import ToTensor
        rgb0 = Image.open('data/panda_static/1.png').resize((256, 256))
        rgb0 = ToTensor()(rgb0).cuda().unsqueeze(0)
        # print('rgb0', rgb0.shape)
        # anim.scheduler.set_timesteps()
        rgb_tensor = torch.randn((1, 4, 16, 32, 32)).cuda() * anim.scheduler.init_noise_sigma
        # rgb_tensor = torch.randn((15, 3, 256, 256)).clamp(0, 1).cuda()
        # rgb_tensor = torch.cat([rgb0.clone()] * 15).cuda()
        # rgb_tensor[0] = rgb0
        rgb_tensor.requires_grad = True
        # optim = torch.optim.AdamW([rgb_tensor], lr=0.05)
        optim = torch.optim.Adam([rgb_tensor], lr=0.01)
        from tqdm import tqdm
        for i in tqdm(range(2000)):
            # rgb_tensor[0] = rgb_tensor[0] * 0 + rgb0
            # loss = anim.train_step(torch.cat([rgb0, rgb_tensor], dim=0), text_emb, as_latent=False)
            loss = anim.train_step(rgb_tensor, text_emb, as_latent=True)
            # loss = anim.train_step(rgb_tensor, text_emb, as_latent=True)
            loss.backward()
            print('grad', rgb_tensor.grad.shape)
            optim.step()
            optim.zero_grad()
            if i % 100 == 0:
                res = anim.decode_latent(rgb_tensor).permute(0, 2, 3, 1)
                # res = torch.cat([rgb0, rgb_tensor], dim=0).permute(0, 2, 3, 1)
                res = res.detach().cpu().numpy()
                res = (res * 255).astype(np.uint8)
                print(res.shape)
                imageio.mimwrite(f'{prefix}_{i}.mp4', res, fps=16, quality=8, macro_block_size=1)
        res = anim.decode_latent(rgb_tensor).permute(0, 2, 3, 1)
        res = rgb_tensor.permute(0, 2, 3, 1)
        res = res.detach().cpu().numpy()
        res = (res * 255).astype(np.uint8)
        print(res.shape)
        imageio.mimwrite(f'{prefix}.mp4', res, fps=16, quality=8, macro_block_size=1)

    # anim.train_step(pred_rgb, text_emb)
