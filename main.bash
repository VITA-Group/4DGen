#optional  image-to-4D data prepocess
#remove background
python preprocess.py --path /data/users/yyy/4DGen_git/4DGen/exp_data/fish.jpg --recenter True

#generate videos by svd
python image_to_video.py --data_path /data/users/yyy/4DGen_git/4DGen/exp_data/fish.jpg_pose0/fish.png --name clown_fish 
#svd results highly rely on random seed. Pick the best result.



cd 4DGen
mkdir data
export name="fish"
export dir="/data/users/yyy/4DGen_git/"
#prepare front image folder. 
# name_pose0
# └── 0.png
# …………
# └── 15.png

#Optional
#if images are not rgba format, run the command below
python preprocess.py --path data/${name}

#generate multi view pseudo labels
cd ..
cd SyncDreamer
#prepare syncdreamer enviroment before (https://github.com/liuyuan-pal/SyncDreamer.git)
#Important!  Move generate_4dgen.py under syncdreamer
python generate_4dgen.py --inp "${dir}/4DGen/data/${name}_pose0" --oup "${dir}/4DGen/data/${name}_sync"

cd ..
cd 4DGen
python preprocess_sync.py --path "data/${name}_sync"



#train your model
#two data files are required
# name_pose0
# └── 0.png
# …………
# └── 15.png

# name_sync
# └── 0_0_0.png
# …………num_seed_view.png
# └── 15_0_15.png
python train.py --configs arguments/i2v.py -e "${name}" --name_override "${name}"


#eval
python python render_for_eval.py --id=${name}  --savedir="${dir}/exp_data/${name}" --model_path='/data/users/yyy/4DGen_git/4DGen/output/2024-03-05/fish_15:09:58'  #please change savedir and model path

export gt_list_data_path="./data/${name}_pose0"
export pred_list_data_path="./exp_data/${name}"
cd evaluation

python evaluation.py  --model 'clip' --gt_list_data_path ${gt_list_data_path} --pred_list_data_path $pred_list_data_path --dataset ${name} \

export input_data_path="${pred_list_data_path}/side/side"
python evaluation.py  --model clip_t --input_data_path ${input_data_path} --dataset $name --direction side --save_name ${name}

input_data_path="${pred_list_data_path}/front/front"
python evaluation.py  --model clip_t --input_data_path $input_data_path --dataset $name --direction front --save_name ${name}

input_data_path="${pred_list_data_path}/back/front"
python evaluation.py  --model clip_t --input_data_path $input_data_path --dataset $name --direction back --save_name ${name}

#xclip
python xclip.py --video_path ./output/fish16_13:50:01/video/ours_3000/multiview.mp4 --prompt a swimming fish

