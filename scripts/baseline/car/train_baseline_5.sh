cd ../../../train 

python train_baseline.py \
	--gpu 0 1 \
	--data-dir /mnt/personal/hutorole/mixoe/data \
	--dataset car \
	--epochs 90 \
	--batch-size 32 \
	--split 1 \
	--run 0