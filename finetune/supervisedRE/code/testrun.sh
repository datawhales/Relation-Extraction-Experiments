seed=42
ckpt_array=("ckpt_triple/entity_marker/ckpt_of_step_5000" "ckpt_triple/entity_marker/ckpt_of_step_10000" "ckpt_triple/entity_marker/ckpt_of_step_15000" "ckpt_triple/entity_marker/ckpt_of_step_20000")
representation="entity_marker"
pool="mean"
prop=0.01
for (( i=0; i<${#ckpt_array[@]}; i++ ))
do
	ckpt=${ckpt_array[i]}
	bash train.sh 1 $seed $ckpt $representation $pool $prop
done