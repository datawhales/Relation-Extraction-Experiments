CUDA_VISIBLE_DEVICES=0 python main.py \
	--seed 42 \
	--lr 3e-5 --batch_size_per_gpu 64 --max_epoch 20 \
	--max_length 100 \
	--mode CM \
	--dataset wiki80 \
	--ckpt_to_load ckpt_exp/end_to_first_concat/ckpt_of_step_60000 \
	--output_representation end_to_first_concat \
	--train_prop 1 \