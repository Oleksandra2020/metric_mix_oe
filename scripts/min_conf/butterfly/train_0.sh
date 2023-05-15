cd ../../../train 

python train_min_conf.py \
	--gpu 0 1 \
	--data-dir /mnt/personal/hutorole/mixoe/data \
	--dataset butterfly \
	--epochs 10 \
	--batch-size 32 \
	--split 0 \
	--run 0 \
	--alpha 1.0 \
	--beta 5.0