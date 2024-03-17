#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams,OptimizationParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def render_set( iteration, views, gaussians, pipeline, background,multiview_video,time_fix=-1,front=False,back=False,side=False,side2=False,id=None,savedir=None):

    render_images = []
    gt_list = []
    render_list = []
    print(len(views))
    
    if multiview_video:
        for idx in tqdm(range (100)):
            view = views[idx]
            if idx == 0:time1 = time()
            if time_fix!=-1:
                ww=torch.tensor([time_fix/16]).unsqueeze(0)
            rendering = render(view['cur_cam'], gaussians, pipeline, background, time=ww, stage='fine')["render"]
            render_images.append(to8b(rendering).transpose(1,2,0))
            render_list.append(rendering)
            save_path=savedir+'/'+id+'/'
            os.makedirs(save_path,exist_ok=True)
            save_path+=str(time_fix)+'.mp4'
    else:        
        for idx in tqdm(range (16)):
            view = views[idx]
            if idx == 0:time1 = time()
            ww = torch.tensor([idx / 16]).unsqueeze(0)
            if front:
                rendering = render(view['front_cam'], gaussians, pipeline, background, time=ww, stage='fine')["render"]
                save_path=savedir+'/'+id+'/front/'
                os.makedirs(save_path,exist_ok=True)
                save_path+='front.mp4'
            if back:
                rendering = render(view['back_cam'], gaussians, pipeline, background, time=ww, stage='fine')["render"]
                save_path=savedir+'/'+id+'/back/'
                os.makedirs(save_path,exist_ok=True)
                save_path+='back.mp4'
            if side:
                rendering = render(view['side_cam'], gaussians, pipeline, background, time=ww, stage='fine')["render"]
                save_path=savedir+'/'+id+'/side/'
                os.makedirs(save_path,exist_ok=True)
                save_path+='side.mp4'
            if side2:
                rendering = render(view['side_cam2'], gaussians, pipeline, background, time=ww, stage='fine')["render"]
                save_path=savedir+'/'+id+'/side2/'
                os.makedirs(save_path,exist_ok=True)
                save_path+='side.mp4'
            render_images.append(to8b(rendering).transpose(1,2,0))
            render_list.append(rendering)

    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))
    print('Len', len(render_images))
    imageio.mimwrite(save_path, render_images, fps=8, quality=8)
    print(save_path)
    
    
def render_sets(dataset : ModelParams, hyperparam, opt,iteration : int, pipeline : PipelineParams,id=None ,savedir=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

         
        for i in range(16):
            render_set(scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,
                multiview_video=True,
                front=False,back=False,side=False,
                time_fix=i,
                id=id,savedir=savedir)
            
        render_set(scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,
                multiview_video=False,
                front=True,back=False,side=False,
                time_fix=-1,
                id=id,savedir=savedir)
        render_set(scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,
                multiview_video=False,
                front=False,back=True,side=False,
                time_fix=-1,
                id=id
                ,savedir=savedir)
        render_set(scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,
                multiview_video=False,
                front=False,back=False,side=True,
                time_fix=-1,
                id=id,savedir=savedir)
        render_set(scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,
            multiview_video=False,
            front=False,back=False,side=False,side2=True,
            time_fix=-1,
            id=id,savedir=savedir)
                
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    # model_path_list=['/data/users/yyy/4dgen_exp/output/2023-11-17/abl_nozero123_11:42:05','/data/users/yyy/4dgen_exp/output/2023-11-18/1wpoint_00:13:03',
    # '/data/users/yyy/4dgen_exp/output/2023-11-16/11.16newbaseline_3_12:08:29','/data/users/yyy/4dgen_exp/output/2023-11-17/no_recon_23:39:40',
    # '/data/users/yyy/4dgen_exp/output/2023-11-17/abl_nopts_05:37:18'
    # ]
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--configs", default='arguments/i2v.py',type=str)
    parser.add_argument("--id", default='wosds',type=str)
    # parser.add_argument("--filename", default='name',type=str)
    parser.add_argument("--savedir", default='./expdata',type=str)
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        options={'ModelParams.name':args.id}
        #print('ModelParams.name:',config.ModelParams.name)
        config.merge_from_dict(options)
        # print('ModelParams.name:',config.ModelParams.name)
        args = merge_hparams(args, config)
        
    print('args:',args.name)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    #id_list=['wosds','1wpoint','wosmooth','worecon','wopts']

    render_sets(model.extract(args), hyperparam.extract(args), op.extract(args),args.iteration, pipeline.extract(args), id=args.name,savedir=args.savedir)