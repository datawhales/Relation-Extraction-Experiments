CUDA_VISIBLE_DEVICES=0 python main.py \
	--seed 42 \
	--lr 3e-5 --batch_size_per_gpu 64 --max_epoch 20 \
	--max_length 100 \
	--mode CM \
	--dataset semeval \
    --ckpt_to_load CP \
	--output_representation entity_marker \
	--train_prop 1 \