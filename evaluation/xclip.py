import torch
import cv2
import os
import numpy as np
from transformers import AutoProcessor, AutoModel
from argparse import ArgumentParser
parser = ArgumentParser(description="Training script parameters")
parser.add_argument('--video_path', type=str)
parser.add_argument('--prompt', type=str)
config = parser.parse_args()

# def read_frames(video_path, num_frames=8):
#     cap = cv2.VideoCapture(video_path)
    
#     frames = []
#     for _ in range(num_frames):
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frames.append(frame)

#     cap.release()
#     return np.array(frames)

# video=read_frames(config.video_path)
# print(video.shape)


images_list = []


for i in range(8):
    image_path = os.path.join(config.video_path, f"{i}.png")
    print('image path:',image_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images_list.append(np.transpose(image, (2, 0, 1)))

video = np.array(images_list, dtype=np.uint8)
print(video.shape)
prompt=config.prompt.replace('_', ' ')
print(prompt)

processor = AutoProcessor.from_pretrained("/data/users/yyy/Largemodel/xclip-base-patch32")

model = AutoModel.from_pretrained("/data/users/yyy/Largemodel/xclip-base-patch32")

inputs = processor(
    text=[prompt],
    videos=list(video),
    return_tensors="pt",
    padding=True,
)
# forward pass
with torch.no_grad():
    outputs = model(**inputs)

logits_per_video = outputs.logits_per_video  # this is the video-text similarity score
#logits_per_video=0
output=f'{config.video_path}    {config.prompt}   logit:{logits_per_video.item()}'
print(output)
save_txt_name = 'xclip_res.txt'
f = open(save_txt_name, 'a+')
f.write(output)
f.write('\n')
f.close()