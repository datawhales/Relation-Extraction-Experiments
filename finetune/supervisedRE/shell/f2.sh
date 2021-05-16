device=1
seed_array=(42 42 42 42 42 42 43 43 43 43 43 43 44 44 44 44 44 44 45 45 45 45 45 45 46 46 46 46 46 46)
batchsize=64
mode="CM"
# param_array=("CM" "CT")
dataset_array=("wiki80" "wiki80" "kbp37" "kbp37" "chemprot" "chemprot" "wiki80" "wiki80" "kbp37" "kbp37" "chemprot" "chemprot" "wiki80" "wiki80" "kbp37" "kbp37" "chemprot" "chemprot" "wiki80" "wiki80" "kbp37" "kbp37" "chemprot" "chemprot" "wiki80" "wiki80" "kbp37" "kbp37" "chemprot" "chemprot" )
# ckpt_array=("EDMAX" "EDMAX" "EDMAX" "EDMAX_5" "EDMAX_5" "EDMAX_5")
ckpt="EDMAX_5"
prop_array=(0.01 0.1 0.01 0.1 0.01 0.1 0.01 0.1 0.01 0.1 0.01 0.1 0.01 0.1 0.01 0.1 0.01 0.1 0.01 0.1 0.01 0.1 0.01 0.1 0.01 0.1 0.01 0.1 0.01 0.1)
log="FINAL_EDMAX_5"
# ckpt="ckpt_final/marker_distance_min_margin5/ckpt_of_step_30000" 
for (( i=0; i<${#seed_array[@]}; i++ ))
do
	seed=${seed_array[i]}
    dataset=${dataset_array[i]}
    prop=${prop_array[i]}
	bash train.sh $device $seed $batchsize $mode $dataset $ckpt $prop $log
done