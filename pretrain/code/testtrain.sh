CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4  main.py \
	--cuda 0,1,2,3 \
        --model TRIPLE \
	--lr 3e-5 --batch_size_per_gpu $1 --max_epoch 20 \
	--gradient_accumulation_steps 1 \
	--max_length 64 \
	--save_step 500 \
	--alpha 0.3 \
	--temperature 0.05 \
	--train_sample \
	--save_dir $2 \
    --margin $3
	--output_representation entity_marker \
    --anchor_feature $4 \