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
import math

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
    
        # load t=0 sequences
        dir=f'data/{self.name}_static_rgba/'
        #dir = 'data/panda_static_rgba/' # generated from new.png
        t0_im_names = [dir + str(x) + '_rgba.png' for x in range(1, 101)]
        # t0_im_names = glob.glob(dir + '/*.png')
        self.t0_im_list = []
        # TODO: should images be RGBA when input??
        for fname in t0_im_names:
            im = Image.open(fname).resize((self.W, self.H))#.convert('RGB')
            # use RGBA
            ww = self.T(im)
            assert ww.shape[0] == 4
            ww[:3] = ww[:3] * ww[-1:] + (1 - ww[-1:])
            self.t0_im_list.append(ww)
        self.t0_im_list = torch.stack(self.t0_im_list, dim=0)#.permute(0, 2, 3, 1)

        print(f"T0 images loaded {self.t0_im_list.shape}.")

        # load pose0 (canonical pose) frames
        # dir = 'data/panda_im/'
        # pose0_im_names = [dir + x for x in ['new.png', '1.png', '2.png', '3.png']]
        #dir = 'data/panda_rgba_pose0/'
        dir=f'data/{self.name}_rgba_pose0/'
        if self.rife==False:
            if frame_num==4:
                if self.name=='panda':
                    frame_list=[0,12,14,15]
                # elif self.name=='rose':
                #     frame_list=[0,6,13,22]
                else:
                    frame_list=range(frame_num)
                pose0_im_names = [dir + f'{x}.png' for x in frame_list]
                # pose0_im_names = [dir + f'frame_{x}_rgba.png' for x in frame_list]
            else:
                if self.name=='astronaut':
                    frame_list= [0] + list(range(12, 27))
                elif self.name=='kitten':
                    frame_list= [0] + list(range(16, 23))+ list(range(24, 32))
                else:
                    frame_list=range(frame_num)
                pose0_im_names = [dir + f'{x}.png' for x in frame_list]
                #pose0_im_names = [dir + f'frame_{x}_rgba.png' for x in range(frame_num)]
        else:
            
            dir=f'data/{self.name}_rife/'
            frame_list=range(frame_num)
            pose0_im_names = [dir + f'img{x}.png' for x in frame_list]
                
        if self.static:
            dir=f'data/{self.name}_rgba_pose0/'
            frame_list=range(frame_num)
            pose0_im_names = [dir + f'{0}.png' for _ in frame_list]
            
        
        # pose0_im_names = pose0_im_names[:2]
        # pose0_im_names = glob.glob(dir + '/*.png')
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
        # self.pose0_im_list = self.pose0_im_list.expand(16, 3, 256, 256)
        print(f"Pose0 images loaded {self.pose0_im_list.shape}")
        self.max_frames = self.pose0_im_list.shape[0]
        print(f"Loaded SDS Dataset. Max {self.max_frames} frames.")

        self.t0_num = self.t0_im_list.shape[0]
        self.pose0_num = self.pose0_im_list.shape[0]
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

        # return Camera(R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
        #                   image_name=f"{index}",uid=index,data_device=torch.device("cuda"),time=time)
        out = {
            # timestamp is per pixel
            "time": time / self.pose0_num,
            't0': self.t0_im_list[t0_idx],
            'pose0': self.pose0_im_list[pose0_idx],
            # 't0': self.t0_im_list[t0_idx].view(-1, 3),
            # 'pose0': self.pose0_im_list[pose0_idx].view(-1, 3),
            # 'bg_color': torch.ones((1, 3), dtype=torch.float32),
            "pose0_cam": self.pose0_cam,
        }

        t0_pose = orbit_camera(0, (t0_idx / self.t0_num) * 360, self.radius)
        ver = 0
        hor = (t0_idx / self.t0_num) * 360
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
        t0_cam = MiniCam(
            t0_pose,
            self.H, # NOTE: order might be wrong
            self.W,
            self.fovy,
            self.fovx,
            self.near,
            self.far,
        )
        out['cur_cam'] = cur_cam
        out['t0_cam'] = t0_cam

        # rand_seq = [t0_cam]
        # start from cur_cam, generate 6 sets of offsets
        # rand_seq = [cur_cam]
        # ver_offset = [np.random.randint(-10, 10) for i in range(self.pose0_num - 1)]
        # hor_offset = [np.random.randint(-10, 10) for i in range(self.pose0_num - 1)]
        # ver_offset = np.cumsum(ver_offset) + ver
        # hor_offset = np.cumsum(hor_offset) + hor
        # ver_offset = np.clip(ver_offset, -15, 45)

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
