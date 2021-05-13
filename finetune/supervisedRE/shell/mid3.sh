device=3
seed=42
batchsize=64
mode="CM"
# param_array=("CM" "CT")
dataset="kbp37"
param_array=("ckpt_final/sentence_length_min/ckpt_of_step_30000" "ckpt_final/sentence_length_min_margin_5/ckpt_of_step_30000")
# param_array=(0.01 0.1 1)
# ckpt="ckpt_final/marker_distance_min_margin5/ckpt_of_step_30000" 
prop=0.1
for (( i=0; i<${#param_array[@]}; i++ ))
do
	ckpt=${param_array[i]}
	bash train.sh $device $seed $batchsize $mode $dataset $ckpt $prop
done