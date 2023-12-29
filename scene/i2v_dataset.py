from torch.utils.data import Dataset
# from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov

from torchvision.transforms import ToTensor
from PIL import Image
import glob
from scene.cam_utils import orbit_camera
import math, os

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar

        w2c = np.linalg.inv(c2w)

        # rectify...
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1

        self.world_view_transform = torch.tensor(w2c).transpose(0, 1)#.cuda()
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            # .cuda()
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(c2w[:3, 3])#.cuda()


class FourDGSdataset(Dataset):
    def __init__(
        self,
        split,
        frame_num = 16,  
        name='panda',
        rife=False,
        static=False,
    ):
        self.split = split
        # self.args = args

        # https://github.com/threestudio-project/threestudio/blob/main/configs/magic123-coarse-sd.yaml#L22
        self.radius = 2.5
        self.W = 512
        self.H = 512
        self.fovy = np.deg2rad(40)
        self.fovx = np.deg2rad(40)
        # self.fovy = np.deg2rad(49.1)
        # self.fovx = np.deg2rad(49.1)
        # align with zero123 rendering setting (ref: https://github.com/cvlab-columbia/zero123/blob/main/objaverse-rendering/scripts/blender_script.py#L61
        self.near = 0.01
        self.far = 100
        self.T = ToTensor()
        self.len_pose0 = frame_num
        self.name=name
        self.rife=rife
        self.static=static
    
        pose0_dir=f'data/{self.name}_pose0/'
        # pose0_dir=f'data/{self.name}_rgba_pose0/'
        
        frame_list = range(frame_num)
        pose0_im_names = [pose0_dir + f'{x}.png' for x in frame_list]
        idx_list = range(frame_num)
        if not os.path.exists(pose0_im_names[0]): # check 0 index
            pose0_im_names = pose0_im_names[1:] + [pose0_dir + f'{frame_num}.png'] # use 1 index
            idx_list = list(idx_list)[1:] + [frame_num]

        base_dir=f'./data/{self.name}_sync'

        syncdreamer_im = []
        # for fname in t0_im_names:
        assert self.static==False
        if self.static==False:
            for frame_idx in idx_list:
            # for frame_idx in range(1, frame_num + 1):
                li = []
                for view_idx in range(16):
                    fname = os.path.join(base_dir, f"{frame_idx}_0_{view_idx}_rgba.png")
                    im = Image.open(fname).resize((self.W, self.H))#.convert('RGB')
                    # use RGBA
                    ww = self.T(im)
                    assert ww.shape[0] == 4
                    ww[:3] = ww[:3] * ww[-1:] + (1 - ww[-1:])
                    li.append(ww)
                li = torch.stack(li, dim=0)#.permute(0, 2, 3, 1)
                syncdreamer_im.append(li)
            self.syncdreamer_im = torch.stack(syncdreamer_im, 0) # [fn, 16, 3, 512, 512]
        else:
            #sync only read frame0
            # (dejia): not used
            for frame_idx in range(frame_num):
                li = []
                frame_idx=0
                for view_idx in range(16):
                    fname = os.path.join(base_dir, f"{frame_idx}_0_{view_idx}_rgba.png")
                    # fname = os.path.join(base_dir, f"{self.name}{frame_idx}_0_{view_idx}_rgba.png")
                    im = Image.open(fname).resize((self.W, self.H))#.convert('RGB')
                    # use RGBA
                    ww = self.T(im)
                    assert ww.shape[0] == 4
                    ww[:3] = ww[:3] * ww[-1:] + (1 - ww[-1:])
                    li.append(ww)
                li = torch.stack(li, dim=0)#.permute(0, 2, 3, 1)
                syncdreamer_im.append(li)
            self.syncdreamer_im = torch.stack(syncdreamer_im, 0) # [fn, 16, 3, 512, 512]

        print(f"syncdreamer images loaded {self.syncdreamer_im.shape}.")

        self.pose0_im_list = []
        # TODO: should images be RGBA when input??
        for fname in pose0_im_names:
            im = Image.open(fname).resize((self.W, self.H))#.convert('RGB')
            ww = self.T(im)
            ww[:3] = ww[:3] * ww[-1:] + (1 - ww[-1:])
            self.pose0_im_list.append(ww)
            # self.pose0_im_list.append(self.T(im))
        while len(self.pose0_im_list) < self.len_pose0:
            self.pose0_im_list.append(ww)
        self.pose0_im_list = torch.stack(self.pose0_im_list, dim=0)#.permute(0, 2, 3, 1)
        # self.pose0_im_list = self.pose0_im_list.expand(fn, 3, 256, 256)
        print(f"Pose0 images loaded {self.pose0_im_list.shape}")
        self.syncdreamer_im = torch.cat([self.pose0_im_list.unsqueeze(1), self.syncdreamer_im], 1)
        print(f"New syncdreamer shape {self.syncdreamer_im.shape}")
        self.max_frames = self.pose0_im_list.shape[0]
        print(f"Loaded SDS Dataset. Max {self.max_frames} frames.")

        # self.t0_num = self.t0_im_list.shape[0]
        self.pose0_num = self.pose0_im_list.shape[0]
        if self.split == 'train':
            self.t0_num = 16 + 1 # fixed
        else:
            self.t0_num = 100
        self.len_ = (self.t0_num) * (self.pose0_num)

        pose0_pose = orbit_camera(0, 0, self.radius)
        self.pose0_cam = MiniCam(
            pose0_pose,
            self.H, # NOTE: order might be wrong
            self.W,
            self.fovy,
            self.fovx,
            self.near,
            self.far,
        )
        self.t0_pose = [self.pose0_cam] + [MiniCam(
        # self.t0_pose = [MiniCam(
            orbit_camera(-30, azimuth, self.radius),
            self.H, # NOTE: order might be wrong
            self.W,
            self.fovy,
            self.fovx,
            self.near,
            self.far,
        ) for azimuth in np.concatenate([np.arange(0, 180, 22.5), np.arange(-180, 0, 22.5)])]
        
        # we sample (pose, t)
    def __getitem__(self, index):
        if self.split == 'train':
            t0_idx = index // self.pose0_num
            pose0_idx = index % self.pose0_num
            time = torch.tensor([pose0_idx]).unsqueeze(0)#.expand(1, self.W * self.H)
        else:
            t0_idx = index # self.t0_num // 2
            pose0_idx = 1
            time = torch.tensor([pose0_idx]).unsqueeze(0)

        out = {
            # timestamp is per pixel
            "time": time / self.pose0_num,
            'pose0': self.pose0_im_list[pose0_idx],
            'pose0_idx': pose0_idx,
            't0_idx': t0_idx,
            't0_weight': min(abs(t0_idx), abs(self.t0_num - t0_idx)),
            # 't0': self.t0_im_list[t0_idx].view(-1, 3),
            # 'pose0': self.pose0_im_list[pose0_idx].view(-1, 3),
            # 'bg_color': torch.ones((1, 3), dtype=torch.float32),
            "pose0_cam": self.pose0_cam,
        }
        #t0_idx=0
        if self.split == 'train':
            out['t0'] = self.syncdreamer_im[0][t0_idx]
            out['gtim'] = self.syncdreamer_im[pose0_idx][t0_idx] # coarse stage

            t0_cam = self.t0_pose[t0_idx]
            out['t0_cam'] = t0_cam
        # out['sync_cam'] = self.sync_pose



        ## for render.py multiview_video

        ver = 0
        hor = (index / 100) * 360
        # ver = np.random.randint(-45, 45)
        # hor = np.random.randint(-180, 180)
        pose = orbit_camera(0 + ver, hor, self.radius)
        out['hor'] = hor
        out['ver'] = ver

        cur_cam = MiniCam(
            pose,
            self.H, # NOTE: order might be wrong
            self.W,
            self.fovy,
            self.fovx,
            self.near,
            self.far,
        )
        out['cur_cam'] = cur_cam 

        # for fine stage, random seq

        rand_seq = []
        ver_list = []
        hor_list = []
        # for i in range(self.pose0_num - 1):
        for i in range(self.pose0_num):
            ver = np.random.randint(-30, 30)
            hor = np.random.randint(-180, 180)
            cur_pose = orbit_camera(ver, hor, self.radius)
            ver_list.append(ver)
            hor_list.append(hor)
            # cur_pose = orbit_camera(ver_offset[i], hor_offset[i], self.radius)
            rand_seq.append(MiniCam(
                cur_pose if self.split == 'train' else pose,
                # cur_pose,
                self.H, # NOTE: order might be wrong
                self.W,
                self.fovy,
                self.fovx,
                self.near,
                self.far,
            ))
        out['rand_poses'] = rand_seq
        out['rand_ver'] = np.array(ver_list)
        out['rand_hor'] = np.array(hor_list)
        # out['rand_ver'] = ver_offset
        # out['rand_hor'] = hor_offset
        
        back_pose=orbit_camera(0, 180, self.radius)
        out['back_cam']=MiniCam(
            back_pose,
            self.H, # NOTE: order might be wrong
            self.W,
            self.fovy,
            self.fovx,
            self.near,
            self.far,
        )
        
        side_pose=orbit_camera(0, 90, self.radius)
        out['side_cam']=MiniCam(
            side_pose,
            self.H, # NOTE: order might be wrong
            self.W,
            self.fovy,
            self.fovx,
            self.near,
            self.far,
        )
        
        side_pose=orbit_camera(0, 70, self.radius)
        out['side_cam2']=MiniCam(
            side_pose,
            self.H, # NOTE: order might be wrong
            self.W,
            self.fovy,
            self.fovx,
            self.near,
            self.far,
        )
        
        front_pose=orbit_camera(0, 0, self.radius)
        out['front_cam']=MiniCam(
            front_pose,
            self.H, # NOTE: order might be wrong
            self.W,
            self.fovy,
            self.fovx,
            self.near,
            self.far,
        )
        return out

    def __len__(self):
        # we sample (pose, t)
        if self.split == 'train':
            return self.len_
        if self.split == 'test':
            return self.pose0_num
            # return self.t0_num
        if self.split == 'video':
            return 100


class ImageDreamdataset(Dataset):
    def __init__(
        self,
        split,
        frame_num = 16,  
        name='panda',
        rife=False,
        static=False,
    ):
        self.split = split
        # self.args = args

        # https://github.com/threestudio-project/threestudio/blob/main/configs/magic123-coarse-sd.yaml#L22
        # self.radius = 2.5
        self.radius = 2.0 ## imagedream https://github.com/bytedance/ImageDream/blob/13e05566ca27c66b6bc5b3ee42bc68ddfb471585/configs/imagedream-sd21-shading.yaml#L20
        self.W = 512
        self.H = 512
        self.fovy = np.deg2rad(40)
        self.fovx = np.deg2rad(40)
        # self.fovy = np.deg2rad(49.1)
        # self.fovx = np.deg2rad(49.1)
        # align with zero123 rendering setting (ref: https://github.com/cvlab-columbia/zero123/blob/main/objaverse-rendering/scripts/blender_script.py#L61
        self.near = 0.01
        self.far = 100
        self.T = ToTensor()
        self.len_pose0 = frame_num
        self.name=name
        self.rife=rife
        self.static=static
    
        pose0_dir=f'./data/ImageDream/{self.name}/rgba/'

        frame_list = range(frame_num)
        pose0_im_names = [pose0_dir + f'{x}.png' for x in frame_list]
        idx_list = range(frame_num)
        if not os.path.exists(pose0_im_names[0]): # check 0 index
            pose0_im_names = pose0_im_names[1:] + [pose0_dir + f'{frame_num}.png'] # use 1 index
            idx_list = list(idx_list)[1:] + [frame_num]

        base_dir=f'./data/output_svd/{self.name}'
        syncdreamer_im = []
        assert self.static==False
        if self.static==False:
            for frame_idx in idx_list:
                li = []
                for view_idx in range(4):
                    #view_idx=0
                    fname = os.path.join(base_dir, f"{frame_idx}_{view_idx}_rgba.png")
                    im = Image.open(fname).resize((self.W, self.H))#.convert('RGB')
                    # use RGBA
                    ww = self.T(im)
                    assert ww.shape[0] == 4
                    ww[:3] = ww[:3] * ww[-1:] + (1 - ww[-1:])
                    li.append(ww)
                li = torch.stack(li, dim=0)#.permute(0, 2, 3, 1)
                syncdreamer_im.append(li)
            self.syncdreamer_im = torch.stack(syncdreamer_im, 0) # [fn, 16, 3, 512, 512]
        else:
            raise NotImplementedError
        



        print(f"imagedream images loaded {self.syncdreamer_im.shape}.")

        self.pose0_im_list = []
        # TODO: should images be RGBA when input??
        for fname in pose0_im_names:
            im = Image.open(fname).resize((self.W, self.H))#.convert('RGB')
            ww = self.T(im)
            ww[:3] = ww[:3] * ww[-1:] + (1 - ww[-1:])
            self.pose0_im_list.append(ww)
            # self.pose0_im_list.append(self.T(im))
        while len(self.pose0_im_list) < self.len_pose0:
            self.pose0_im_list.append(ww)
        self.pose0_im_list = torch.stack(self.pose0_im_list, dim=0)#.permute(0, 2, 3, 1)
        # self.pose0_im_list = self.pose0_im_list.expand(fn, 3, 256, 256)
        print(f"Pose0 images loaded {self.pose0_im_list.shape}")
        # self.syncdreamer_im = torch.cat([self.pose0_im_list.unsqueeze(1), self.syncdreamer_im], 1)
        print(f"New syncdreamer shape {self.syncdreamer_im.shape}")
        self.max_frames = self.pose0_im_list.shape[0]
        print(f"Loaded SDS Dataset. Max {self.max_frames} frames.")

        # self.t0_num = self.t0_im_list.shape[0]
        self.pose0_num = self.pose0_im_list.shape[0]
        if self.split == 'train':
            self.t0_num = 4# + 1 # fixed
        else:
            self.t0_num = 100
        self.len_ = (self.t0_num) * (self.pose0_num)

        # NOTE: this is different!!
        pose0_pose = orbit_camera(0, 90, self.radius)
        self.pose0_cam = MiniCam(
            pose0_pose,
            self.H, # NOTE: order might be wrong
            self.W,
            self.fovy,
            self.fovx,
            self.near,
            self.far,
        )
        # self.t0_pose = [self.pose0_cam] + [MiniCam(
        self.t0_pose = [MiniCam(
            orbit_camera(0, azimuth, self.radius),
            self.H, # NOTE: order might be wrong
            self.W,
            self.fovy,
            self.fovx,
            self.near,
            self.far,
        ) for azimuth in np.concatenate([np.arange(0, 180, 90), np.arange(-180, 0, 90)])]
        
        # we sample (pose, t)
    def __getitem__(self, index):
        if self.split == 'train':
            t0_idx = index // self.pose0_num
            pose0_idx = index % self.pose0_num
            time = torch.tensor([pose0_idx]).unsqueeze(0)#.expand(1, self.W * self.H)
        else:
            t0_idx = index # self.t0_num // 2
            pose0_idx = 1
            time = torch.tensor([pose0_idx]).unsqueeze(0)

        out = {
            # timestamp is per pixel
            "time": time / self.pose0_num,
            'pose0': self.pose0_im_list[pose0_idx],
            'pose0_idx': pose0_idx,
            't0_idx': t0_idx,
            't0_weight': min(abs(t0_idx), abs(self.t0_num - t0_idx)),
            # 't0': self.t0_im_list[t0_idx].view(-1, 3),
            # 'pose0': self.pose0_im_list[pose0_idx].view(-1, 3),
            # 'bg_color': torch.ones((1, 3), dtype=torch.float32),
            "pose0_cam": self.pose0_cam,
        }
        #t0_idx=0
        if self.split == 'train':
            out['t0'] = self.syncdreamer_im[0][t0_idx]
            out['gtim'] = self.syncdreamer_im[pose0_idx][t0_idx] # coarse stage

            t0_cam = self.t0_pose[t0_idx]
            out['t0_cam'] = t0_cam

        ## for render.py multiview_video
        ver = 0
        hor = (index / 100) * 360
        pose = orbit_camera(0 + ver, hor, self.radius)
        out['hor'] = hor
        out['ver'] = ver

        cur_cam = MiniCam(
            pose,
            self.H, # NOTE: order might be wrong
            self.W,
            self.fovy,
            self.fovx,
            self.near,
            self.far,
        )
        out['cur_cam'] = cur_cam 

        # for fine stage, random seq

        rand_seq = []
        ver_list = []
        hor_list = []
        # for i in range(self.pose0_num - 1):
        for i in range(self.pose0_num):
            ver = np.random.randint(-30, 30)
            hor = np.random.randint(-180, 180)
            cur_pose = orbit_camera(ver, hor, self.radius)
            ver_list.append(ver)
            hor_list.append(hor)
            # cur_pose = orbit_camera(ver_offset[i], hor_offset[i], self.radius)
            rand_seq.append(MiniCam(
                cur_pose if self.split == 'train' else pose,
                # cur_pose,
                self.H, # NOTE: order might be wrong
                self.W,
                self.fovy,
                self.fovx,
                self.near,
                self.far,
            ))
        out['rand_poses'] = rand_seq
        out['rand_ver'] = np.array(ver_list)
        out['rand_hor'] = np.array(hor_list)
        # out['rand_ver'] = ver_offset
        # out['rand_hor'] = hor_offset
        
        back_pose=orbit_camera(0, 180, self.radius)
        out['back_cam']=MiniCam(
            back_pose,
            self.H, # NOTE: order might be wrong
            self.W,
            self.fovy,
            self.fovx,
            self.near,
            self.far,
        )
        
        side_pose=orbit_camera(0, 90, self.radius)
        out['side_cam']=MiniCam(
            side_pose,
            self.H, # NOTE: order might be wrong
            self.W,
            self.fovy,
            self.fovx,
            self.near,
            self.far,
        )
        
        side_pose=orbit_camera(0, 70, self.radius)
        out['side_cam2']=MiniCam(
            side_pose,
            self.H, # NOTE: order might be wrong
            self.W,
            self.fovy,
            self.fovx,
            self.near,
            self.far,
        )
        
        front_pose=orbit_camera(0, 0, self.radius)
        out['front_cam']=MiniCam(
            front_pose,
            self.H, # NOTE: order might be wrong
            self.W,
            self.fovy,
            self.fovx,
            self.near,
            self.far,
        )

        ver = np.random.randint(-30, 30)
        hor = np.random.randint(-180, 180)
        li = [orbit_camera(ver, hor, self.radius)]
        for view_i in range(1, 4):
            li.append(orbit_camera(ver, hor + 90 * view_i, self.radius))
        out['dream_pose_mat'] = torch.from_numpy(np.stack(li, axis=0))
        out['dream_pose'] = [MiniCam(
                cur_pose,
                # cur_pose,
                self.H, # NOTE: order might be wrong
                self.W,
                self.fovy,
                self.fovx,
                self.near,
                self.far,
            ) for cur_pose in li]
        return out

    def __len__(self):
        # we sample (pose, t)
        if self.split == 'train':
            return self.len_
        if self.split == 'test':
            return self.pose0_num
            # return self.t0_num
        if self.split == 'video':
            return 100
