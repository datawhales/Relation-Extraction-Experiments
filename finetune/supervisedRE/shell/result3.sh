device=2
seed_array=(42 42 42 43 43 43 44 44 44 45 45 45 46 46 46)
batchsize=64
mode="CM"
# param_array=("CM" "CT")
# dataset="wiki80"
dataset_array=("wiki80" "kbp37" "chemprot" "wiki80" "kbp37" "chemprot" "wiki80" "kbp37" "chemprot" "wiki80" "kbp37" "chemprot" "wiki80" "kbp37" "chemprot")
# ckpt_array=("EDMAX" "EDMAX" "EDMAX" "EDMAX_5" "EDMAX_5" "EDMAX_5")
ckpt="EDMIN_5"
prop=1
log="EDMIN_5_whole"
# ckpt="ckpt_final/marker_distance_min_margin5/ckpt_of_step_30000" 
for (( i=0; i<${#seed_array[@]}; i++ ))
do
	seed=${seed_array[i]}
    dataset=${dataset_array[i]}
	bash train.sh $device $seed $batchsize $mode $dataset $ckpt $prop $log
done