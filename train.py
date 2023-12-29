

import numpy as np
import random
import os
import torch

from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer

import lpips
import gc
from torchvision import transforms as T
from utils.scene_utils import render_training_image
from time import time
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from guidance.zero123_utils import Zero123

from PIL import Image
from torchvision.transforms import ToTensor
from kaolin.metrics.pointcloud import chamfer_distance
from plyfile import PlyData

def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter,timer, args):
    first_iter = 0


    torch.cuda.empty_cache()
    gc.collect()
    print(f'Start training of stage {stage}: ')
    zero123 = Zero123('cuda')
    dir=f'data/{args.name}_pose0/'
    # dir=f'data/{args.name}_rgba_pose0/'
    # if args.i2v:
    #     frame_list=range(1, 1 + args.frame_num)
    # e1lse:
    frame_list = range(args.frame_num)
    pose0_im_names = [dir + f'{x}.png' for x in frame_list]
    if not os.path.exists(pose0_im_names[0]): # check 0 index
        pose0_im_names = pose0_im_names[1:] + [dir + f'{args.frame_num}.png'] # use 1 index
    print('pose0_im_names:',pose0_im_names)
    T = ToTensor()
    im_list = []
    for fname in pose0_im_names:
        im = Image.open(fname).resize((512, 512))
        ww = T(im)
        assert ww.shape[0] == 4
        ww[:3] = ww[:3] * ww[-1:] + (1 - ww[-1:])
        im_list.append((ww))
    pose0_im = torch.stack(im_list).cuda().detach()
    print('pose0_im shape:',pose0_im.shape)
    pose0_embed1, pose0_embed2 = zero123.get_img_embeds_pil(pose0_im[:,:3, :, :] , pose0_im[:,:3, :, :] )
    print('pose0_embed1 shape:',pose0_embed1.shape)
    print('pose0_embed2 shape:',pose0_embed2.shape)
    stage_ = ['static', 'coarse', 'fine']
    train_iter_ = [opt.static_iterations, opt.coarse_iterations, opt.iterations]
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda", requires_grad=False)
    lpips_model = lpips.LPIPS(net="alex").cuda()
    for cur_stage, train_iter in zip(stage_, train_iter_):
        gaussians.training_setup(opt)
        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)
        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)
        viewpoint_stack = None
        ema_loss_for_log = 0.0
        ema_psnr_for_log = 0.0

        final_iter = train_iter

        progress_bar = tqdm(range(first_iter, final_iter), desc=f"[{args.expname}] Training progress")
        video_cams = scene.getVideoCameras()
        for iteration in range(first_iter, final_iter+1):
            stage = cur_stage
            loss_weight = 1

            iter_start.record()
            gaussians.update_learning_rate(iteration)
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras()
                viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=1,shuffle=True,num_workers=4,collate_fn=list)
                frame_num = viewpoint_stack.pose0_num

                loader = iter(viewpoint_stack_loader)
            if True:
                try:
                    data = next(loader)
                except StopIteration:
                    print("reset dataloader")
                    batch_size = 1
                    loader = iter(viewpoint_stack_loader)
            if (iteration - 1) == debug_from:
                pipe.debug = True
            images = []
            gt_images = []
            radii_list = []
            visibility_filter_list = []
            viewspace_point_tensor_list = []
            dx = []
            ds = []
            dr = []
            do = []
            dc=[]
            out_pts = []
            if stage in ['static']:
                viewpoint_cam = data[0]['t0_cam']
                if stage == 'static':
                    render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage, time=0)
                else:
                    raise NotImplementedError
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                rgba = torch.cat([image, render_pkg['alpha']], dim=0)
                images.append(rgba.unsqueeze(0))
                # gt_image = data[0]['gtim'].to(image.device)
                gt_image = data[0]['t0'].to(image.device)
                if data[0]['t0_idx'] == 0:
                    loss_weight = 10
                # gt_image = data[0]['gtim'].to(image.device)
                gt_images.append(gt_image.unsqueeze(0))
                radii_list.append(radii.unsqueeze(0))
                visibility_filter_list.append(visibility_filter.unsqueeze(0))
                viewspace_point_tensor_list.append(viewspace_point_tensor)
            elif stage == 'coarse':
                for i in range(1):
                    time = data[0]['time']
                    viewpoint_cam = data[0]['t0_cam']
                    render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage, time=time, return_pts=True)
                    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                    means3D = render_pkg['means3D']
                    rgba = torch.cat([image, render_pkg['alpha']], dim=0)
                    images.append(rgba.unsqueeze(0))
                    gt_image = data[0]['gtim'].to(image.device)
                    gt_images.append(gt_image.unsqueeze(0))
                    if data[0]['t0_idx'] == 0:
                        loss_weight = 10
                    out_pts.append(means3D)
                    if 'dx' in render_pkg:
                        dx.append(render_pkg['dx'])
                        ds.append(render_pkg['ds'])
                        dr.append(render_pkg['dr'])
                        do.append(render_pkg['do'])
                        dc.append(render_pkg['dc'])
                    radii_list.append(radii.unsqueeze(0))
                    visibility_filter_list.append(visibility_filter.unsqueeze(0))
                    viewspace_point_tensor_list.append(viewspace_point_tensor)
            else:
                rand_seed=np.random.random()
                if  rand_seed< args.fine_rand_rate:
                    viewpoint_cam = data[0]['rand_poses']
                    fps = 1 / frame_num
                    set_t0_frame0 = True
                    t0 = 0
                    if frame_num > 16:
                        sds_idx_list = np.random.choice(range(frame_num), 16)
                    else:
                        sds_idx_list = range(frame_num)
                    # for i in range(frame_num):
                    for i in sds_idx_list:
                        time = torch.tensor([t0 + i * fps]).unsqueeze(0).float()
                        render_pkg = render(viewpoint_cam[i], gaussians, pipe, background, stage=stage, time=time)
                        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                        fg_mask = render_pkg['alpha']
                        rgba = torch.cat([image, fg_mask], dim=0)
                        images.append(rgba.unsqueeze(0))
                        if 'dx' in render_pkg:
                            dx.append(render_pkg['dx'])
                            ds.append(render_pkg['ds'])
                            dr.append(render_pkg['dr'])
                            do.append(render_pkg['do'])
                            dc.append(render_pkg['dc'])
                        radii_list.append(radii.unsqueeze(0))
                        visibility_filter_list.append(visibility_filter.unsqueeze(0))
                        viewspace_point_tensor_list.append(viewspace_point_tensor)
                else:
                    for i in range(1):
                        time = data[0]['time']
                        viewpoint_cam = data[0]['t0_cam']
                        render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage, time=time, return_pts=True)
                        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                        means3D = render_pkg['means3D']
                        rgba = torch.cat([image, render_pkg['alpha']], dim=0)
                        images.append(rgba.unsqueeze(0))
                        gt_image = data[0]['gtim'].to(image.device)
                        gt_images.append(gt_image.unsqueeze(0))
                        if data[0]['t0_idx'] == 0:
                            loss_weight = 10
                        out_pts.append(means3D)
                        if 'dx' in render_pkg:
                            dx.append(render_pkg['dx'])
                            ds.append(render_pkg['ds'])
                            dr.append(render_pkg['dr'])
                            do.append(render_pkg['do'])
                            dc.append(render_pkg['dc'])
                        radii_list.append(radii.unsqueeze(0))
                        visibility_filter_list.append(visibility_filter.unsqueeze(0))
                        viewspace_point_tensor_list.append(viewspace_point_tensor)
            radii = torch.cat(radii_list,0).max(dim=0).values
            visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
            image_tensor = torch.cat(images,0)
            if len(out_pts):
                out_pts = torch.stack(out_pts,0)
            use_zero123 = True
            use_animate = False

            if len(gt_images):
                gt_image_tensor = torch.cat(gt_images,0)
            if stage in ['static']:
                Ll1 = l1_loss(image_tensor, gt_image_tensor)
                tb_writer.add_scalar(f'{stage}/loss_recon', Ll1.item(), iteration)
                lpipsloss = lpips_loss(image_tensor, gt_image_tensor,lpips_model)
                tb_writer.add_scalar(f'{stage}/loss_lpips', lpipsloss.item(), iteration)
                loss = Ll1 * 10 + lpipsloss * 20


            elif stage == 'coarse':
                Ll1 = l1_loss(image_tensor, gt_image_tensor)
                tb_writer.add_scalar(f'{stage}/loss_recon', Ll1.item(), iteration)
                lpipsloss = lpips_loss(image_tensor, gt_image_tensor,lpips_model)
                tb_writer.add_scalar(f'{stage}/loss_lpips', lpipsloss.item(), iteration)
                loss = Ll1 * 10 + lpipsloss * 20


                time_now=int(time.item()*args.frame_num)
            else:
                if rand_seed < args.fine_rand_rate:
                    if use_zero123:
                        loss = 0
                        loss_zero123_total=0

                        for idx in sds_idx_list:
                        # for idx in range(0, frame_num):
                            cur_emb = (pose0_embed1[idx].unsqueeze(0), pose0_embed2[idx])
                            loss_zero123, im = zero123.train_step(image_tensor[idx:idx+1,:3, :, :], data[0]['rand_ver'][idx], data[0]['rand_hor'][idx], 0, cur_emb)
                            loss_zero123_total += loss_zero123
                        tb_writer.add_scalar(f'{stage}/loss_zero123', loss_zero123_total.item(), iteration)
                        loss += loss_zero123_total / len(sds_idx_list) * args.lambda_zero123
                else:
                    Ll1 = l1_loss(image_tensor, gt_image_tensor)
                    tb_writer.add_scalar(f'{stage}/loss_recon', Ll1.item(), iteration)
                    lpipsloss = lpips_loss(image_tensor,gt_image_tensor,lpips_model)
                    tb_writer.add_scalar(f'{stage}/loss_lpips', lpipsloss.item(), iteration)
                    loss = Ll1 * 10 + lpipsloss * 20

                    time_now=int(time.item()*args.frame_num)

            loss = loss * loss_weight

            if stage == "fine" and hyper.time_smoothness_weight != 0:
                tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.plane_tv_weight, hyper.l1_time_planes)
                loss += tv_loss
                tb_writer.add_scalar(f'{stage}/loss_tv', tv_loss.item(), iteration)
            if opt.lambda_dssim != 0 and len(gt_images) != 0:
                ssim_loss = 1 - ssim(image_tensor,gt_image_tensor)
                loss += opt.lambda_dssim * (ssim_loss)
                tb_writer.add_scalar(f'{stage}/loss_ssim', ssim_loss.item(), iteration)
            if opt.lambda_lpips != 0 and len(gt_images) != 0:
                lpipsloss = lpips_loss(image_tensor,gt_image_tensor,lpips_model)
                loss += opt.lambda_lpips * lpipsloss
                tb_writer.add_scalar(f'{stage}/loss_lpips', lpipsloss.item(), iteration)
            # if len(dx)!=1 and len(dx)!=0:
            #     loss_dx_tv = torch.stack([x if i else x.detach() for i, x in enumerate(dx[:-1])]) - torch.stack(dx[1:])
            loss.backward()
            viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
            for idx in range(0, len(viewspace_point_tensor_list)):
                if viewspace_point_tensor_list[idx].grad is not None:
                    viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
            iter_end.record()
            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

                total_point = gaussians._xyz.shape[0]
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                            "point":f"{total_point}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()
                timer.pause()
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, pipe, background, stage)
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration, stage)
                if dataset.render_process:
                    if (iteration < 1000 and iteration % 10 == 1) \
                        or (iteration < 3000 and iteration % 50 == 1) \
                            or (iteration < 10000 and iteration %  100 == 1) \
                                or (iteration < 60000 and iteration % 100 ==1):
                        render_training_image(scene, gaussians, video_cams, render, pipe, background, stage, iteration-1,timer.get_elapsed_time())
                timer.start()
                if stage == 'static' and iteration < opt.densify_until_iter:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)
                    if stage in ['static', "coarse"]:
                        opacity_threshold = opt.opacity_threshold_coarse
                        densify_threshold = opt.densify_grad_threshold_coarse
                    else:
                        opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)
                        densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter)
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        print('>>>>>>> Now densify')
                        gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    pruning_interval = opt.pruning_interval if stage != 'fine' else opt.pruning_interval_fine
                    if iteration > opt.pruning_from_iter and iteration % pruning_interval == 0:
                        print('>>>>>>> Now pruning', opt.opacity_reset_interval)
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                        gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname, args):
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians,load_coarse=None)
    timer.start()
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                             checkpoint_iterations, checkpoint, debug_from,
                             gaussians, scene, "coarse", tb_writer, opt.coarse_iterations,timer, args)

from datetime import datetime

def prepare_output_and_logger(expname):
    if not args.model_path:
        unique_str = str(datetime.today().strftime('%Y-%m-%d')) + '/' + expname + '_' + datetime.today().strftime('%H:%M:%S')
        args.model_path = os.path.join("./output/", unique_str)
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, pipe, bg, stage):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
    ww = iteration if stage == 'static' else iteration
    if iteration % 100 == 0 and ww in testing_iterations:
        torch.cuda.empty_cache()
        train_set = scene.getTrainCameras()
        validation_configs = [{'name': 'train', 'cameras' : [train_set[idx % len(train_set)] for idx in range(10, 5000, 299)]}]
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ti = (torch.tensor([0]).unsqueeze(0))
                cam_li = config['cameras'][0]['rand_poses']
                im_li = []
                num = len(cam_li)
                for tii in range(num):
                    if stage == 'static':
                        ti = (torch.tensor([tii * 0]).unsqueeze(0).cuda())
                    else:
                        ti = (torch.tensor([tii / num]).unsqueeze(0).cuda())
                    viewpoint = cam_li[tii]
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, pipe=pipe, bg_color=bg, time=ti)["render"], 0.0, 1.0)
                    im_li.append(image)
                ww = len(im_li) // 2
                r1 = torch.cat(im_li[:ww], dim=-1)
                r2 = torch.cat(im_li[ww:], dim=-1)
                im_li = torch.cat([r1, r2], dim=-2)
                if tb_writer:
                    tb_writer.add_image(f"rand_seq/{stage}", im_li, global_step=iteration)
                for idx, data in enumerate(config['cameras']):
                    if stage == 'static':
                        ti = (torch.tensor([0]).unsqueeze(0).cuda())
                        viewpoint = data['t0_cam']
                    else:
                        ti = data['time']
                        viewpoint = data['t0_cam']
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, pipe=pipe, bg_color=bg, time=ti)["render"], 0.0, 1.0)
                    if stage == 'static':
                        gt_image = data['gtim'][:3].to(image.device)
                    elif stage == 'coarse':
                        gt_image = data['gtim'][:3].to(image.device)
                    else:
                        gt_image = data['gtim'][:3].to(image.device)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(stage + "/{}/render".format(idx), image[None], global_step=iteration)
                        tb_writer.add_images(stage + "/{}/gt".format(idx), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum()/scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram(f"{stage}/scene/motion_histogram", scene.gaussians._deformation_accum.mean(dim=-1)/100, iteration,max_bins=500)
        torch.cuda.empty_cache()
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i*50 for i in range(0,300)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2000,2500, 3000, 5000, 7_000, 8000, 9000, 14000, 20000, 30_000,45000,60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('-e', "--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "arguments/ours/i2v_xdj.py")
    parser.add_argument("--yyypath", type=str, default = "")
    parser.add_argument("--t0_frame0_rate", type=float, default = 1)
    parser.add_argument("--name_override", type=str, default="")
    parser.add_argument("--sds_ratio_override", type=float, default=-1)
    parser.add_argument("--sds_weight_override", type=float, default=-1)
    parser.add_argument("--iteration", default=-1, type=int)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations - 1)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    if args.name_override != '':
        args.name = args.name_override
    if args.sds_ratio_override != -1:
        args.fine_rand_rate = args.sds_ratio_override
    if args.sds_weight_override != -1:
        args.lambda_zero123 = args.sds_weight_override
    # print(args.name)
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    timer1 = Timer()
    timer1.start()
    print('Configs: ', args)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname, args)
    print("\nTraining complete.")
    print('training time:',timer1.get_elapsed_time())
    from render import render_sets

    render_sets(lp.extract(args), hp.extract(args), op.extract(args), args.iterations, pp.extract(args), skip_train=True, skip_test=True, skip_video=False, multiview_video=True)
    print("\Rendering complete.")
