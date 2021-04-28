CUDA_VISIBLE_DEVICES=0 python main.py \
	--seed 42 \
	--lr 3e-5 --batch_size_per_gpu 64 --max_epoch 20 \
	--max_length 100 \
	--mode CM \
	--dataset wiki80 \
	--ckpt_to_load ckpt_TS/entity_marker/ckpt_of_step_60000 \
	--output_representation entity_marker \
	--train_prop 1 \