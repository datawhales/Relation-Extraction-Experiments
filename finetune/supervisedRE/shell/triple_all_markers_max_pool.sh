CUDA_VISIBLE_DEVICES=0 python main.py \
	--seed 42 \
	--lr 3e-5 --batch_size_per_gpu 64 --max_epoch 20 \
	--max_length 100 \
	--mode CM \
	--dataset wiki80 \
	--ckpt_to_load ckpt_triple/all_markers_max/ckpt_of_step_60000 \
	--output_representation all_markers \
    --pooling_method max \
	--train_prop 1 \