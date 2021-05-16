device=0
seed=42
batchsize=64
mode="CM"
# param_array=("CM" "CT")
dataset="wiki80"
ckpt_array=("EDMAX" "EDMAX" "EDMAX" "EDMAX_5" "EDMAX_5" "EDMAX_5")
prop_array=(0.01 0.1 1 0.01 0.1 1)
# ckpt="ckpt_final/marker_distance_min_margin5/ckpt_of_step_30000" 
for (( i=0; i<${#ckpt_array[@]}; i++ ))
do
	ckpt=${ckpt_array[i]}
    prop=${prop_array[i]}
	bash train.sh $device $seed $batchsize $mode $dataset $ckpt $prop
done