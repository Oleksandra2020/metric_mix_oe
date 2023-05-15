cd ../../../train 

python train_noise.py \
	--gpu 0 1 \
	--data-dir /mnt/personal/hutorole/mixoe/data \
	--dataset car \
	--epochs 10 \
	--batch-size 32 \
	--split 1 \
	--run 0 \
	--alpha 1.0 \
	--beta 5.0