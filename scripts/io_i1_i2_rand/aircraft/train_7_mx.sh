cd ../../../train 

python train_triplet_io_i1_i2_rand.py \
	--gpu 0 1 \
	--data-dir /mnt/personal/hutorole/mixoe/data \
	--dataset aircraft \
	--epochs 10 \
	--batch-size 32 \
	--beta 0.3 \
	--beta2 5.0 \
	--alpha 1.0 \
	--margin 0.5 \
	--mixup 2 \
	--id 1