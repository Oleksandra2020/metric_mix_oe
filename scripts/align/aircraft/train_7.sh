cd ../../../train 

python train_align.py \
	--gpu 0 1 \
	--data-dir /mnt/personal/hutorole/mixoe/data \
	--dataset aircraft \
	--epochs 10 \
	--batch-size 32 \
	--split 1 \
	--run 2 \
	--alpha 1.0 \
	--beta 5.0