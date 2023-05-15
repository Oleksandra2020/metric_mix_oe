cd ../../../train 

python train_mixoe_manifold.py \
	--gpu 0 1 \
	--data-dir /mnt/personal/hutorole/mixoe/data \
	--dataset butterfly \
	--epochs 100 \
	--batch-size 32 \
	--split 0 \
	--run 4 \
	--alpha 1.0 \
	--beta 5.0