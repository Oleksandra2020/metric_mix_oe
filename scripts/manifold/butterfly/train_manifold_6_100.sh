cd ../../../train 

python train_mixoe_manifold.py \
	--gpu 0 1 \
	--data-dir /mnt/personal/hutorole/mixoe/data \
	--dataset butterfly \
	--epochs 100 \
	--batch-size 32 \
	--split 1 \
	--run 1 \
	--alpha 1.0 \
	--beta 5.0