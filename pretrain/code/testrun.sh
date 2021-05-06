savedir_array=("ckpt_triple/entity_marker_random_anchor_margin_1.0" "ckpt_triple/entity_marker_random_anchor_margin_1.5" "ckpt_triple/entity_marker_random_anchor_margin_2.0")
margin_array=(0.5 1.0 1.5 2.0)
anchorfeature="marker_dist"
batchsize=64
for (( i=0; i<${#savedir_array[@]}; i++ ))
do
	savedir=${savedir_array[i]}
    margin=${margin_array[i]}
	bash testtrain.sh $batchsize $savedir $margin $anchorfeature
done