import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import cv2
# import clip
from transformers import CLIPFeatureExtractor, CLIPModel, CLIPTokenizer
from torchvision import transforms
from argparse import ArgumentParser
import torch.nn.functional as F
import glob
import os,sys
from argparse import ArgumentParser, Namespace


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    # print(x.shape, y.shape)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

class CLIP(nn.Module):
    def __init__(self, device, clip_name = 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K'):
        super().__init__()

        self.device = device

        clip_name = clip_name

        self.feature_extractor = CLIPFeatureExtractor.from_pretrained(clip_name)
        self.clip_model = CLIPModel.from_pretrained(clip_name).cuda()
        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
    
    
        self.normalize = transforms.Normalize(mean=self.feature_extractor.image_mean, std=self.feature_extractor.image_std)

        self.resize = transforms.Resize(224)

         # image augmentation
        self.aug = T.Compose([
            T.Resize((224, 224)),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    
    def get_text_embeds(self, prompt, neg_prompt=None, dir=None):

        clip_text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.cuda()
        text_z = self.clip_model.get_text_features(clip_text_input)
        # text = clip.tokenize(prompt).to(self.device)
        # text_z = self.clip_model.encode_text(text)
        text_z = text_z / text_z.norm(dim=-1, keepdim=True)

        return text_z

    def set_epoch(self, epoch):
        pass

    def get_img_embeds(self, img):
        img = self.aug(img)
        image_z = self.clip_model.get_image_features(img)
        image_z = image_z / image_z.norm(dim=-1, keepdim=True) # normalize features
        return image_z

    
    def train_step(self, text_z, pred_rgb, image_ref_clip, **kwargs):

        pred_rgb = self.resize(pred_rgb)
        pred_rgb = self.normalize(pred_rgb)

        image_z = self.clip_model.get_image_features(pred_rgb)
        image_z = image_z / image_z.norm(dim=-1, keepdim=True) # normalize features

        # print(image_z.shape, text_z.shape)
        loss = spherical_dist_loss(image_z, image_ref_clip)

        # loss = - (image_z * text_z).sum(-1).mean()

        return loss
    
    def text_loss(self, text_z, pred_rgb):

        pred_rgb = self.resize(pred_rgb)
        pred_rgb = self.normalize(pred_rgb)

        image_z = self.clip_model.get_image_features(pred_rgb)
        image_z = image_z / image_z.norm(dim=-1, keepdim=True) # normalize features

        # print(image_z.shape, text_z.shape)
        loss = spherical_dist_loss(image_z, text_z)

        # loss = - (image_z * text_z).sum(-1).mean()

        return loss
    
    def img_loss(self, img_ref_z, pred_rgb):
        # pred_rgb = self.aug(pred_rgb)
        pred_rgb = self.resize(pred_rgb)
        pred_rgb = self.normalize(pred_rgb)       
        image_z = self.clip_model.get_image_features(pred_rgb)
        image_z = image_z / image_z.norm(dim=-1, keepdim=True) # normalize features
        #print(image_z.shape)
        
        img_ref_z = self.resize(img_ref_z)
        img_ref_z = self.normalize(img_ref_z)
        img_ref_z = self.clip_model.get_image_features(img_ref_z)
        img_ref_z = img_ref_z / img_ref_z.norm(dim=-1, keepdim=True) # normalize features
        
        # loss = - (image_z * img_ref_z).sum(-1).mean()
        loss = spherical_dist_loss(image_z, img_ref_z)

        return loss

    def single_image(self,gtpath,path):
        gt_pose0_path=gtpath
        pred_pose0_path=path
        pred=readimage(pred_pose0_path)
        gt=readimage(gt_pose0_path)
        loss=self.img_loss(gt,pred)
        #print('loss:',loss.item())
        return loss.item()

def readimage(path):
    image=torch.from_numpy(cv2.imread(path)/255).permute(2,0,1).unsqueeze(0).float().cuda()
    return image


def get_subdirectories(folder_path):
    subdirectories = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    return subdirectories


class Eval():
    def __init__(self):
        super().__init__()
        self.clip=CLIP('cuda')
    
    def CLIP_T(self,input_data_path,name=None,direction=None):
        #input:data path, includes n images
        input_data=glob.glob(f'{input_data_path}/*.png')
        input_data=sorted(input_data,key=lambda info: (int(info.split('/')[-1].split('.')[0])))
        
        loss_total=0
        for i in range(1,len(input_data)):
            data=input_data[i]
            data_pre=input_data[i-1]
            data=readimage(data)
            data_pre=readimage(data_pre)
            loss=self.clip.img_loss(data_pre,data).item()
            loss_total+=loss
        clip_t_loss=loss_total/(len(input_data)-1)
        if name!=None:
            print('Dataset:',name,"   direction:",direction,"   clip_t:",clip_t_loss)
            save_data='Dataset:'+name+"   direction:"+direction+"   clip_t:"+str(clip_t_loss)+'\n'
            with open('/home/yyy/data/4dgen_exp_pl/4dgen_exp/CLIP_Loss/output.txt', 'a+') as file:
                file.write(save_data)
        else:
            print("clip_t:",clip_t_loss)
    
    
            
    def CLIP_(self,gt_list_data_path,pred_list_data_path,name=None):
        #input:
        #gt_list_data_path, file path includes n frames
        #pred_list_data_path,file path includes n files, each file include m pose images
        
        #for example: 
        #gt_list_data_path/0.png, gt_list_data_path/number.png
        #pred_list_data_path/0/0.png, pred_list_data_path/number/posenumber.png
        gt_data=glob.glob(f'{gt_list_data_path}/*.png')
        gt_data=sorted(gt_data,key=lambda info: (int(info.split('/')[-1].split('.')[0])))
        len_gt=len(gt_data)
        
        loss_all_frame=0
        for i in range(16):
            pred_path=pred_list_data_path+'/'+str(i)
            pred_data=glob.glob(f'{pred_path}/*.png')
            pred_data=sorted(pred_data,key=lambda info: (int(info.split('/')[-1].split('.')[0])))
            len_pred=len(pred_data)
            
            loss_one_frame=0
            for j in range(len_pred):
                loss=self.clip.single_image(gt_data[i],pred_data[j])
                loss_one_frame+=loss
            loss_one_frame_avg=loss_one_frame/len_pred
            loss_all_frame+=loss_one_frame_avg
        loss_all_frame_avg=loss_all_frame/16
        
        if name!=None:
            print('Datset:',name,"   clip:",loss_all_frame_avg)
            save_data='Datset:'+name+"   clip:"+str(loss_all_frame_avg)+'\n'
            with open('/home/yyy/data/4dgen_exp_pl/4dgen_exp/CLIP_Loss/output.txt', 'a+') as file:
                file.write(save_data)
            
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model",default='clip', type=str)
    parser.add_argument("--direction",default='front', type=str)
    parser.add_argument("--dataset",default='rose', type=str)
    parser.add_argument("--gt_list_data_path",default='rose', type=str)
    parser.add_argument("--pred_list_data_path",default='rose', type=str)
    parser.add_argument("--input_data_path",default='rose', type=str)
    args = parser.parse_args()

    eval=Eval()
    if args.model=='clip':
        eval.CLIP_(args.gt_list_data_path,args.pred_list_data_path,args.dataset)
    elif args.model=='clip_t':
        eval.CLIP_T(args.input_data_path,args.dataset,args.direction)
        
        

        
        
