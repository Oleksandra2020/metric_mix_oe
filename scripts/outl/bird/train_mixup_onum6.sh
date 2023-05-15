cd ../../train 

python train_mixoe_outl.py \
	--gpu 0 1 \
	--data-dir /mnt/personal/hutorole/mixoe/data \
	--dataset bird \
	--epochs 10 \
	--batch-size 32 \
	--mix-op mixup \
	--alpha 1.0 \
	--beta 5.0 \
	--outlier_num 20000 \
	--outlier_classes 0