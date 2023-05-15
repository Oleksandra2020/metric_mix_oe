cd ../../../train 

python train_mixoe.py \
	--gpu 0 1 \
	--data-dir /mnt/personal/hutorole/mixoe/data \
	--dataset aircraft \
	--epochs 10 \
	--batch-size 32 \
	--mix-op mixup \
	--split 2 \
	--run 1 \
	--alpha 1.0 \
	--beta 5.0