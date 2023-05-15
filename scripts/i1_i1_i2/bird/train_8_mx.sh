cd ../../../train 

python train_triplet_i1_i1_i2.py \
	--gpu 0 1 \
	--data-dir /mnt/personal/hutorole/mixoe/data \
	--dataset bird \
	--epochs 10 \
	--batch-size 32 \
	--beta 0.5 \
	--beta2 5.0 \
	--alpha 1.0 \
	--margin 0.5 \
	--mixup 2 \
	--id 1