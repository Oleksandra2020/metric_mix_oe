cd ../../../train 

python train_triplet_io_i1_o_rand.py \
	--gpu 0 1 \
	--data-dir /mnt/personal/hutorole/mixoe/data \
	--dataset car \
	--epochs 10 \
	--batch-size 32 \
	--beta 0.3 \
	--alpha 1.0 \
	--margin 0.5 \
	--mixup 0 \
	--id 1