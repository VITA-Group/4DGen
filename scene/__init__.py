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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
# from scene.dataset import FourDGSdataset
from scene.i2v_dataset import FourDGSdataset, ImageDreamdataset
# from scene.rife_sync_dataset import FourDGSdataset
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.utils.data import Dataset
import numpy as np

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None,shuffle=True, resolution_scales=[1.0], load_coarse=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}
        self.cameras_extent = 1 # scene_info.nerf_normalization["radius"]

        print("Loading Training Cameras")
        if args.imagedream:
            ds = ImageDreamdataset
        else:
            ds = FourDGSdataset
        print('args.frame_num:',args.frame_num)
        self.train_camera = ds(split='train', frame_num=args.frame_num,name=args.name,rife=args.rife,static=args.static)
        print("Loading Test Cameras")
        self.maxtime = self.train_camera.pose0_num
        self.test_camera = ds(split='test', frame_num=args.frame_num,name=args.name,rife=args.rife,static=args.static)
        print("Loading Video Cameras")
        self.video_cameras = ds(split='video', frame_num=args.frame_num,name=args.name,rife=args.rife,static=args.static)
        xyz_max = [2.5, 2.5, 2.5]
        xyz_min = [-2.5, -2.5, -2.5]
        self.gaussians._deformation.deformation_net.grid.set_aabb(xyz_max,xyz_min)
        # assert not self.loaded_iter
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians.load_model(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                   ))
        else:
            # TODO: accept argparse
            num_pts = int(2.5e4)
            

            # random init
            self.gaussians.random_init(num_pts, 1, radius=0.5)
            # point cloud init
            
            # cloud_path='./data/eagle1_1.ply' # 
                        
            # 4 is not used
            # self.gaussians.load_3studio_ply(cloud_path, spatial_lr_scale=1, time_line=self.maxtime, step=1, position_scale=1, load_color=False) ## imagedream

    def save(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))

        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)
    def getTrainCameras(self, scale=1.0):
        return self.train_camera

    def getTestCameras(self, scale=1.0):
        return self.test_camera
    def getVideoCameras(self, scale=1.0):
        return self.video_cameras