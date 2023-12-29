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
def render_set(model_path, name, iteration, views, gaussians, pipeline, background,multiview_video, fname='video_rgb.mp4'):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_list = []
    render_list = []
    print(len(views))
    
    # for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
    # for idx in tqdm(range (100)):
    fnum = 100
    # fnum = 12
    for idx in tqdm(range (fnum)):
        view = views[idx]
        if idx == 0:time1 = time()
        #ww = torch.tensor([idx / 12]).unsqueeze(0)
        ww = torch.tensor([idx / fnum]).unsqueeze(0)
        # ww = torch.tensor([idx / 100]).unsqueeze(0)

        if multiview_video:
            rendering = render(view['cur_cam'], gaussians, pipeline, background, time=ww, stage='fine')["render"]
        else:
            rendering = render(view['pose0_cam'], gaussians, pipeline, background, time=ww, stage='fine')["render"]
        render_images.append(to8b(rendering).transpose(1,2,0))
        render_list.append(rendering)
    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))
    print('Len', len(render_images))
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), fname), render_images, fps=8, quality=8)
    
    
def render_set_timefix(model_path, name, iteration, views, gaussians, pipeline, background,multiview_video, fname='video_rgb.mp4',time_fix=-1):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_list = []
    render_list = []
    print(len(views))
    
    # for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
    for idx in tqdm(range (12)):
    #for idx in tqdm(range (100)):
        view = views[idx]
        if idx == 0:time1 = time()
        # ww = torch.tensor([idx / 16]).unsqueeze(0)
        ww = torch.tensor([idx / 100]).unsqueeze(0)
        if time_fix!=-1:
            ww=torch.tensor([time_fix/16]).unsqueeze(0)
        if multiview_video:
            rendering = render(view['cur_cam'], gaussians, pipeline, background, time=ww, stage='fine')["render"]

        render_images.append(to8b(rendering).transpose(1,2,0))
        render_list.append(rendering)
    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))
    print('Len', len(render_images))
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), fname), render_images, fps=7, quality=8)
    
    
def render_sets(dataset : ModelParams, hyperparam, opt,iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool,multiview_video: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,multiview_video)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,multiview_video)
        if not skip_video:
            #origin
            render_set(dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,multiview_video=True, fname='multiview.mp4')
            render_set(dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,multiview_video=False, fname='pose0.mp4')
                
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument('--multiview_video',default=False,action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), hyperparam.extract(args), op.extract(args),args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video,args.multiview_video)
