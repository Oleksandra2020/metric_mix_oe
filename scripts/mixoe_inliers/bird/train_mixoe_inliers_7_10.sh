cd ../../../train 

python train_mixoe_inliers.py \
	--gpu 0 1 \
	--data-dir /mnt/personal/hutorole/mixoe/data \
	--dataset bird \
	--epochs 10 \
	--batch-size 32 \
	--split 1 \
	--run 2 \
	--alpha 1.0 \
	--beta 5.0