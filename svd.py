import os
import random
from concurrent.futures import ThreadPoolExecutor
directory_path = '/data/users/yyy/4dgen_exp/data/harmonyview_testset'

# 获取目录下所有文件
file_list = os.listdir(directory_path)

# 打印文件列表
# for file in file_list:
#     file_name=file.split('/')[-1].split('.')[0]
#     file_name='harm_'+file_name
#     print(file_name)
#     file_path=directory_path+'/'+file
#     cmd=f'CUDA_VISIBLE_DEVICES="1" python image_to_video.py --data_path {file_path} --name {file_name}'
#     print(cmd)
#     os.system(cmd)

def process_file(file):
    file_name = file.split('/')[-1].split('.')[0]
    file_name = 'harm_' + file_name
    print(file_name)
    
    file_path = os.path.join(directory_path, file)
    cuda_device = random.randint(0, 4)
    cmd = f'CUDA_VISIBLE_DEVICES="{cuda_device}" python image_to_video.py --data_path {file_path} --name {file_name}'
    print(cmd)
    
    os.system(cmd)
    
for file in file_list:
    with ThreadPoolExecutor() as executor:
        executor.map(process_file, file_list)