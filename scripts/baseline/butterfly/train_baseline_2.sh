cd ../../../train 

python train_baseline.py \
	--gpu 0 1 \
	--data-dir /mnt/personal/hutorole/mixoe/data \
	--dataset butterfly \
	--epochs 90 \
	--batch-size 32 \
	--split 0 \
	--run 2