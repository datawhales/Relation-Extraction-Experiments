CUDA_VISIBLE_DEVICES=$1 python main.py \
	--seed $2 \
	--lr 3e-5 --batch_size_per_gpu 64 --max_epoch 20 \
	--max_length 100 \
	--mode CM \
	--dataset wiki80 \
    --ckpt_to_load $3 \
    --output_representation $4 \
    --pooling_method $5 \
	--train_prop $6 \