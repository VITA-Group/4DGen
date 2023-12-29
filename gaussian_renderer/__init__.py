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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, time=torch.tensor([[0]]), scaling_modifier = 1.0, override_color = None, stage=None, render_flow=False, return_pts=False):
    # print(scaling_modifier)
    assert scaling_modifier == 1
    if stage is None:
        raise NotImplementedError
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception as e:
        # print(e)
        pass

    # Set up rasterization configuration
    
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation
    means3D = pc.get_xyz
    try:
        assert time.item() >= 0 and time.item() <= 1
        time = time.to(means3D.device).repeat(means3D.shape[0],1)
    except:
        assert time >= 0 and time <= 1
        time = torch.tensor([time]).to(means3D.device).repeat(means3D.shape[0],1)
    # time = time / 16 # in range of [0, 1]

    means2D = screenspace_points
    opacity = pc._opacity
    color=pc._features_dc
    color=color[:,0,:]
    
    
    
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    dx = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # scales = pc.get_scaling
        scales = pc._scaling
        if scales.shape[-1] == 1:
            scales = scales.repeat(1, 3)
        #scales = torch.ones_like(scales ) * 0.03
        # rotations = pc.get_rotation
        rotations = pc._rotation
    deformation_point = pc._deformation_table
    # print('color render:',color.shape)   #[40000, 1, 3]->[40000, 3]
    # print('rotations render:',rotations.shape)  #[40000, 4]
    
    if stage == "static": # or time.sum() == 0:
    # if stage == "static" or time.sum() == 0:
        means3D_deform, scales_deform, rotations_deform, opacity_deform,color_deform = means3D, scales, rotations, opacity,color
    elif stage in ["coarse", 'fine']:
        means3D_deform, scales_deform, rotations_deform, opacity_deform, color_deform = pc._deformation(means3D[deformation_point], scales[deformation_point], rotations[deformation_point], opacity[deformation_point], color[deformation_point], time[deformation_point])
        dx = (means3D_deform - means3D[deformation_point])
        ds = (scales_deform - scales[deformation_point])
        dr = (rotations_deform - rotations[deformation_point])
        do = (opacity_deform - opacity[deformation_point])
        dc = (color_deform - color[deformation_point])
    else:
        # deprecated
        means3D_deform, scales_deform, rotations_deform, opacity_deform,color_deform = pc._deformation(means3D[deformation_point].detach(), scales[deformation_point].detach(), rotations[deformation_point].detach(), opacity[deformation_point].detach(),color[deformation_point].detach(), time[deformation_point].detach())
        dx = (means3D_deform - means3D[deformation_point].detach())
        ds = (scales_deform - scales[deformation_point].detach())
        dr = (rotations_deform - rotations[deformation_point].detach())
        do = (opacity_deform - opacity[deformation_point].detach())
        #dc=0
        dc=(color_deform - color[deformation_point].detach())

    means3D_final = torch.zeros_like(means3D)
    rotations_final = torch.zeros_like(rotations)
    scales_final = torch.zeros_like(scales)
    opacity_final = torch.zeros_like(opacity)
    color_final= torch.zeros_like(color)
    means3D_final[deformation_point] =  means3D_deform
    rotations_final[deformation_point] =  rotations_deform
    scales_final[deformation_point] =  scales_deform
    opacity_final[deformation_point] = opacity_deform
    
    # print('color_final shape before',color_final.shape)

    # print('color_final shape',color_final.shape)
    # print('color_deform shape',color_deform.shape)
    # print('deformation_point shape',deformation_point.shape)
    color_final[deformation_point] = color_deform

    means3D_final[~deformation_point] = means3D[~deformation_point]
    rotations_final[~deformation_point] = rotations[~deformation_point]
    scales_final[~deformation_point] = scales[~deformation_point]
    opacity_final[~deformation_point] = opacity[~deformation_point]
    color_final[~deformation_point] = color[~deformation_point]
    color_final=torch.unsqueeze(color_final, 1)  #[40000,  3]->[40000, 1, 3]
    
    scales_final = pc.scaling_activation(scales_final)
    #scales_final = torch.ones_like(scales_final ) * 0.01
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity)
    #color without activation
    
    # print(opacity.max())
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    
    #pc._features_dc=color_final  #update color
    
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            # print('shs=============')
            #shs = pc.get_features
            # dc=pc.get_features_dc
            # print('pc.get_features_dc devide',pc.get_features_dc.device)
            dc=color_final
            #print('color_final devide',dc.device)
            rest=pc.get_features_rest
            shs=torch.cat((dc, rest), dim=1)
    else:
        colors_precomp = override_color
    
    #colors_precomp=color_final    #not sure
    # print('colors_precomp shape:',colors_precomp.shape)
    # print('color_final shape:',color_final.shape)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth, alpha = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    res = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
        "alpha": alpha,
        "depth":depth,
    }
    # print(dx, time.sum(), stage)
    if dx is not None:
        res['dx'] = dx #.mean()
        res['ds'] = ds #.mean()
        res['dr'] = dr #.mean()
        res['do'] = do #.mean()
        res['dc'] = dc

    if render_flow and stage == 'coarse':
        flow_screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            flow_screenspace_points.retain_grad()
        except:
            pass
        rendered_flow, _, _, _ = rasterizer(
            means3D = means3D_final,
            means2D = flow_screenspace_points,
            shs = None,
            colors_precomp = dx,
            opacities = opacity,
            scales = scales_final,
            rotations = rotations_final,
            cov3D_precomp = cov3D_precomp
        )
        res['rendered_flow'] = rendered_flow
    if return_pts:
        res['means3D'] = means3D_final
        res['means2D'] = means2D
        res['opacity_final'] = opacity_final
    return res

