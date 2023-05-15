cd ../../../train 

python train_mixoe.py \
	--gpu 0 1 \
	--data-dir /mnt/personal/hutorole/mixoe/data \
	--dataset butterfly \
	--epochs 10 \
	--batch-size 32 \
	--mix-op mixup \
	--split 0 \
	--run 4 \
	--alpha 1.0 \
	--beta 5.0