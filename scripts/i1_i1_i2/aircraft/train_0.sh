cd ../../../train 

python train_triplet_i1_i1_i2.py \
	--gpu 0 1 \
	--data-dir /mnt/personal/hutorole/mixoe/data \
	--dataset aircraft \
	--epochs 10 \
	--batch-size 32 \
	--beta 0.1 \
	--alpha 1.0 \
	--margin 0.05 \
	--mixup 0 \
	--id 1