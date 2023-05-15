cd ../../../train 

python train_label.py \
	--gpu 0 1 \
	--data-dir /mnt/personal/hutorole/mixoe/data \
	--dataset car \
	--epochs 10 \
	--batch-size 32 \
	--split 0 \
	--run 3 \
	--alpha 1.0 \
	--beta 5.0