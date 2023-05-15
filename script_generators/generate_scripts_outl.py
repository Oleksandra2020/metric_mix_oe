import os


def main(data_dir, alpha, beta):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    mixup_variants = ['mixup']

    outlier_nums = [50, 100, 250, 500, 1000, 10000, 20000]
    outlier_classes = [0, 1, 5, 10, 15, 50, 100, 500]
    epochs = 10

    dataset = data_dir.split('/')[-1]

    if dataset == 'air':
        dataset = 'aircraft'


    for mx in mixup_variants:

        j = 0
        for outlier_num in outlier_nums:

            script_skeleton = f"cd ../../train \n\npython train_mixoe_outl.py \\\n\t--gpu 0 1 \\\n\t--data-dir /mnt/personal/hutorole/mixoe/data \\\n\t--dataset {dataset} \\\n\t--epochs {epochs} \\\n\t--batch-size 32 \\\n\t--mix-op {mx} \\\n\t--alpha {alpha} \\\n\t--beta {beta} \\\n\t--outlier_num {outlier_num} \\\n\t--outlier_classes 0"
            file_name = f"train_{mx}_onum{j}.sh"

            with open(os.path.join(data_dir, file_name), "w+") as f:
                f.write(script_skeleton)

            j += 1


        for outlier_class in outlier_classes:

            script_skeleton = f"cd ../../train \n\npython train_mixoe_outl.py \\\n\t--gpu 0 1 \\\n\t--data-dir /mnt/personal/hutorole/mixoe/data \\\n\t--dataset {dataset} \\\n\t--epochs {epochs} \\\n\t--batch-size 32 \\\n\t--mix-op {mx} \\\n\t--alpha {alpha} \\\n\t--beta {beta} \\\n\t--outlier_num 0 \\\n\t--outlier_classes {outlier_class}"
            file_name = f"train_{mx}_oclass{outlier_class}.sh"

            with open(os.path.join(data_dir, file_name), "w+") as f:
                f.write(script_skeleton)


if __name__ == "__main__":
    data_dir = "./scripts/outl/aircraft"
    alpha, beta = 1.0, 5.0

    main(data_dir, alpha, beta)

    data_dir = "./scripts/outl/bird"
    alpha, beta = 1.0, 5.0

    main(data_dir, alpha, beta)


    data_dir = "./scripts/outl/car"
    alpha, beta = 1.0, 5.0

    main(data_dir, alpha, beta)

    data_dir = "./scripts/outl/butterfly"
    alpha, beta = 1.0, 5.0

    main(data_dir, alpha, beta)
