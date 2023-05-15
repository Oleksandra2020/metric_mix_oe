cd ../../../train 

python train_triplet_i1_i1_o.py \
	--gpu 0 1 \
	--data-dir /mnt/personal/hutorole/mixoe/data \
	--dataset butterfly \
	--epochs 10 \
	--batch-size 32 \
	--beta 0.5 \
	--alpha 1.0 \
	--margin 0.05 \
	--mixup 0 \
	--id 1