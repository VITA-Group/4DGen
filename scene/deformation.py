import functools
import math
import os
import time
# from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import torch.nn.init as init
from collections import OrderedDict
from scene.hexplane import HexPlaneField

class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, skips=[], args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.grid_merge = args.grid_merge

        self.no_grid = args.no_grid
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires, grid_merge=args.grid_merge)
        self.pos_deform, self.scales_deform, self.rotations_deform, self.opacity_deform ,self.color_deform= self.create_net()
        # self.pos_deform.fc1.weight.data.zero_()
        # self.pos_deform.fc1.bias.data.zero_()
        # self.scales_deform.fc1.weight.data.zero_()
        # self.scales_deform.fc1.bias.data.zero_()
        # self.rotations_deform.fc1.weight.data.zero_()
        # self.rotations_deform.fc1.bias.data.zero_()
        # self.opacity_deform.fc1.weight.data.zero_()
        # self.opacity_deform.fc1.bias.data.zero_()
        # self.color_deform.fc1.weight.data.zero_()
        # self.color_deform.fc1.bias.data.zero_()

        self.args = args
    def create_net(self):
        
        mlp_out_dim = 0
        if self.no_grid:
            self.feature_out = [nn.Linear(4,self.W)]
        else:
            if self.grid_merge == 'cat':
                self.feature_out = [nn.Linear(mlp_out_dim + self.grid.feat_dim * 6, self.W)]
            else:
                self.feature_out = [nn.Linear(mlp_out_dim + self.grid.feat_dim, self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        output_dim = self.W
        # pose, scale, rotation, opacity
        return  \
            nn.Sequential(
                OrderedDict([
                    ('act0', nn.ReLU()),
                    ('fc2', nn.Linear(self.W, self.W)),
                    ('act3', nn.ReLU()),
                    ('fc1', nn.Linear(self.W, 3)),
                ])
                # nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)
            ),\
            nn.Sequential(
                OrderedDict([
                    ('act0', nn.ReLU()),
                    ('fc2', nn.Linear(self.W, self.W)),
                    ('act3', nn.ReLU()),
                    ('fc1', nn.Linear(self.W, 1)),
                ])
                # nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1)
            ),\
            nn.Sequential(
                OrderedDict([
                    ('act0', nn.ReLU()),
                    ('fc2', nn.Linear(self.W, self.W)),
                    ('act3', nn.ReLU()),
                    ('fc1', nn.Linear(self.W, 4)),
                ])
                # nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4)
            ), \
            nn.Sequential(
                OrderedDict([
                    ('act0', nn.ReLU()),
                    ('fc2', nn.Linear(self.W, self.W)),
                    ('act3', nn.ReLU()),
                    ('fc1', nn.Linear(self.W, 1)),
                ])
                # nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1)
            ),\
            nn.Sequential(
                OrderedDict([
                    ('act0', nn.ReLU()),
                    ('fc2', nn.Linear(self.W, self.W)),
                    ('act3', nn.ReLU()),
                    ('fc1', nn.Linear(self.W, 3)),
                    ('act4',nn.Tanh())
                ])
                # nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1)
            )
    
    def query_time(self, rays_pts_emb, scales_emb, rotations_emb, time_emb):

        if self.no_grid:
            h = torch.cat([rays_pts_emb[:,:3],time_emb[:,:1]],-1)
        else:
            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])

            h = grid_feature
        
        h = self.feature_out(h)
  
        return h

    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None,color=None, time_emb=None):
        # if time_emb.sum() == 0: 
        # # if time_emb is None:
        #     return self.forward_static(rays_pts_emb[:,:3], scales_emb, rotations_emb, opacity, time_emb)
        #     # return self.forward_static(rays_pts_emb[:,:3])
        # else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, color,time_emb)

    def forward_static(self, pts, scales, rotations, opacity, time):
    # def forward_static(self, rays_pts_emb):
        return pts, scales, rotations, opacity
        # print('??????? forward_static')
        # grid_feature = self.grid(rays_pts_emb[:,:3])
        # dx = self.static_mlp(grid_feature)
        # return rays_pts_emb[:, :3] + dx
    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb,color_emb, time_emb):
        hidden = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_emb).float()
        dx = self.pos_deform(hidden)
        pts = rays_pts_emb[:, :3] + dx
        # print(scales_emb.shape, rotations_emb.shape, opacity_emb.shape, time_emb.shape)
        # print('no_ds', self.args.no_ds, self.args.no_dr, self.args.no_do)
        if self.args.no_ds:
            scales = scales_emb[:,:3]
        else:
            ds = self.scales_deform(hidden)
            scales = scales_emb[:,:3] + ds
            
        if self.args.no_dr:
            rotations = rotations_emb[:,:4]
        else:
            #print('dr======================================')
            dr = self.rotations_deform(hidden)   #[40000, 4]
            rotations = rotations_emb[:,:4] + dr   #([40000, 3]+[40000, 4]=[40000, 4]
            # print('rotations_emb[:,:3] shape===',rotations_emb[:,:3].shape)
            # print('dr shape=======',dr.shape)
            # print('rotations shape=======',rotations.shape)
            
        if self.args.no_do:
            opacity = opacity_emb[:,:1] 
        else:
            do = self.opacity_deform(hidden) 
            opacity = opacity_emb[:,:1] + do
            
        if self.args.no_dc:
            # print('no dc======================================')
            color=color_emb[:,:3]
        else:
            # print('dc======================================')
            # print('hidden shape=======',hidden.shape)
            dc = self.color_deform(hidden)   #[40000, 256]->[40000, 3]
            color = color_emb[:,:3] + dc    #[40000, 3]+[40000, 3]
            # print('color_emb[:,:3] shape===',color_emb[:,:3].shape)
            # print('dc shape=======',dc.shape)
            # print('color shape=======',color.shape)
            # hidden shape======= torch.Size([40000, 256]) [11/11 13:55:16]
            # color_emb[:,:3] shape=== torch.Size([40000, 1, 3]) [11/11 13:55:16]
            # dc shape======= torch.Size([40000, 3]) [11/11 13:55:16]
            # color shape======= torch.Size([40000, 40000, 3]) [11/11 13:55:16]
            # color_final shape torch.Size([40000, 1, 3]) [11/11 13:55:16]
            # color_deform shape torch.Size([40000, 40000, 3]) [11/11 13:55:16]
            # deformation_point shape torch.Size([40000]) [11/11 13:55:16]
        # + do
        # print("deformation value:","pts:",torch.abs(dx).mean(),"rotation:",torch.abs(dr).mean())

        return pts, scales, rotations, opacity,color
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    def get_grid_parameters(self):
        return list(self.grid.parameters() ) 
    # + list(self.timegrid.parameters())
class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        times_ch = 2*timebase_pe+1
        # self.timenet = nn.Sequential(
        # nn.Linear(times_ch, timenet_width), nn.ReLU(),
        # nn.Linear(timenet_width, timenet_output))
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(4+3)+((4+3)*scale_rotation_pe)*2, input_ch_time=timenet_output, args=args)
        # self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)
        # print(self)

    def forward(self, point, scales=None, rotations=None, opacity=None,color=None, times_sel=None):
        # raise NotImplementedError
        # print('>>>>> time', times_sel)
        if times_sel is not None:
            means3D_, scales_, rotations_, opacity_ ,color_= self.forward_dynamic(point, scales, rotations, opacity,color, times_sel)
            # return means3D_, scales, rotations, opacity
            return means3D_, scales_, rotations_, opacity_,color_
        else:
            raise NotImplementedError
            return self.forward_static(point)

        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None,color=None, times_sel=None):
        # times_emb = poc_fre(times_sel, self.time_poc)

        means3D, scales, rotations, opacity,color = self.deformation_net( point,
                                                  scales,
                                                rotations,
                                                opacity,
                                                color,
                                                # times_feature,
                                                times_sel)
        return means3D, scales, rotations, opacity,color
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters()
        # + list(self.timenet.parameters())
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

def initialize_weights(m):
    pass
    # if isinstance(m, nn.Linear):
    #     init.constant_(m.weight, 0)
    #     # init.xavier_uniform_(m.weight,gain=1)
    #     if m.bias is not None:
    #         # init.xavier_uniform_(m.weight,gain=1)
    #         init.constant_(m.bias, 0)
