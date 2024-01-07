#change paths befor running
clip_gt_list_data_path=(
    '/home/yyy/data/4dgen_exp_pl/4dgen_exp/data/rose_rgba_pose0'
    '/home/yyy/data/4dgen_exp_pl/4dgen_exp/data/fish_rgba_pose0'
    '/home/yyy/data/4dgen_exp_pl/4dgen_exp/data/tulip_rgba_pose0'
    '/home/yyy/data/4dgen_exp_pl/4dgen_exp/data/ironman_rgba_pose0'
)

clip_pred_list_data_path=(
    '/home/yyy/data/4dgen_exp_pl/4dgen_exp/exp_data/eval/ours/rose'
    '/home/yyy/data/4dgen_exp_pl/4dgen_exp/exp_data/eval/ours/fish'
    '/home/yyy/data/4dgen_exp_pl/4dgen_exp/exp_data/eval/ours/tulip'
    '/home/yyy/data/4dgen_exp_pl/4dgen_exp/exp_data/eval/ours/ironman'
)

clip_12345_pred_list_data_path=(
    '/home/yyy/data/4dgen_exp_pl/4dgen_exp/exp_data/eval/12345/rose'
    '/home/yyy/data/4dgen_exp_pl/4dgen_exp/exp_data/eval/12345/fish'
    '/home/yyy/data/4dgen_exp_pl/4dgen_exp/exp_data/eval/12345/tulip'
    '/home/yyy/data/4dgen_exp_pl/4dgen_exp/exp_data/eval/12345/ironman'
)

clip_dr_pred_list_data_path=(
    '/home/yyy/data/4dgen_exp_pl/4dgen_exp/exp_data/eval/dr/rose'
    '/home/yyy/data/4dgen_exp_pl/4dgen_exp/exp_data/eval/dr/fish'
    '/home/yyy/data/4dgen_exp_pl/4dgen_exp/exp_data/eval/dr/tulip'
    '/home/yyy/data/4dgen_exp_pl/4dgen_exp/exp_data/eval/dr/ironman'
)



Dataset=(
    'rose'
    'fish'
    'tulip'
    'ironman'
)

Dataset_dr=(
    'rose_dr'
    'fish_dr'
    'tulip_dr'
    'ironman_dr'
)

Dataset_12345=(
    'rose_12345'
    'fish_12345'
    'tulip_12345'
    'ironman_12345'
)


for ((i = 0; i < ${#clip_gt_list_data_path[@]}; i++)); do
    gt_list_data_path="${clip_gt_list_data_path[$i]}"
    pred_list_data_path="${clip_pred_list_data_path[$i]}"
    dataset="${Dataset[$i]}"

    python evaluation.py  --model 'clip' --gt_list_data_path $gt_list_data_path --pred_list_data_path $pred_list_data_path --dataset $dataset
done


for ((i = 0; i < ${#clip_gt_list_data_path[@]}; i++)); do
    gt_list_data_path="${clip_gt_list_data_path[$i]}"
    pred_list_data_path="${clip_12345_pred_list_data_path[$i]}"
    dataset="${Dataset_12345[$i]}"

    python evaluation.py  --model 'clip' --gt_list_data_path $gt_list_data_path --pred_list_data_path $pred_list_data_path --dataset $dataset
done

for ((i = 0; i < ${#clip_gt_list_data_path[@]}; i++)); do
    gt_list_data_path="${clip_gt_list_data_path[$i]}"
    pred_list_data_path="${clip_dr_pred_list_data_path[$i]}"
    dataset="${Dataset_dr[$i]}"

    python evaluation.py  --model 'clip' --gt_list_data_path $gt_list_data_path --pred_list_data_path $pred_list_data_path --dataset $dataset
done



# #clip_t
frontstr='/front'
backstr='/back'
sidestr='/side'
for ((i = 0; i < ${#clip_gt_list_data_path[@]}; i++)); do
    input_data_path="${clip_pred_list_data_path[$i]}$frontstr"
    echo $input_data_path
    dataset="${Dataset[$i]}"
    python evaluation.py  --model clip_t --input_data_path $input_data_path --dataset $dataset --direction front

    input_data_path="${clip_pred_list_data_path[$i]}$backstr"
    dataset="${Dataset[$i]}"
    python evaluation.py  --model clip_t --input_data_path $input_data_path --dataset $dataset --direction back

    input_data_path="${clip_pred_list_data_path[$i]}$sidestr"
    dataset="${Dataset[$i]}"
    python evaluation.py  --model clip_t --input_data_path $input_data_path --dataset $dataset --direction side


done

for ((i = 0; i < ${#clip_gt_list_data_path[@]}; i++)); do
    input_data_path="${clip_12345_pred_list_data_path[$i]}$frontstr"
    dataset="${Dataset_12345[$i]}"
    python evaluation.py  --model clip_t --input_data_path $input_data_path --dataset $dataset --direction front

    input_data_path="${clip_12345_pred_list_data_path[$i]}$backstr"
    dataset="${Dataset_12345[$i]}"
    python evaluation.py  --model clip_t --input_data_path $input_data_path --dataset $dataset --direction back

    input_data_path="${clip_12345_pred_list_data_path[$i]}$sidestr"
    dataset="${Dataset_12345[$i]}"
    python evaluation.py  --model clip_t --input_data_path $input_data_path --dataset $dataset --direction side


done

for ((i = 0; i < ${#clip_gt_list_data_path[@]}; i++)); do
    input_data_path="${clip_dr_pred_list_data_path[$i]}$frontstr"
    dataset="${Dataset_dr[$i]}"
    python evaluation.py  --model clip_t --input_data_path $input_data_path --dataset $dataset --direction front

    input_data_path="${clip_dr_pred_list_data_path[$i]}$backstr"
    dataset="${Dataset_dr[$i]}"
    python evaluation.py  --model clip_t --input_data_path $input_data_path --dataset $dataset --direction back

    input_data_path="${clip_dr_pred_list_data_path[$i]}$sidestr"
    dataset="${Dataset_dr[$i]}"
    python evaluation.py  --model clip_t --input_data_path $input_data_path --dataset $dataset --direction side


done

