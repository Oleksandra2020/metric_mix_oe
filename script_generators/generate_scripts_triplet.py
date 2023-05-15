import os


def main(data_dir, alphay, beta):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    dataset = data_dir.split('/')[-1]

    if dataset == 'air':
        dataset = 'aircraft'

    margins = [0.05, 0.1, 0.5, 1.]
    betas = [0.1, 0.3, 0.5]
    epochs = 10

    # script_skeleton = f"cd ../../../train \n\npython train_baseline.py \\\n\t--gpu 0 1 \\\n\t--data-dir /mnt/personal/hutorole/mixoe/data \\\n\t--dataset {dataset} \\\n\t--epochs {epochs} \\\n\t--batch-size 32"
    # file_name = f"train_baseline.sh"

    # with open(os.path.join(data_dir, file_name), "w+") as f:
    #     f.write(script_skeleton)

    i = 0
    for margin in margins:

        for beta in betas:

            # script_skeleton = f"cd ../../../train \n\npython train_triplet_io_i1_i2_rand.py \\\n\t--gpu 0 1 \\\n\t--data-dir /mnt/personal/hutorole/mixoe/data \\\n\t--dataset {dataset} \\\n\t--epochs {epochs} \\\n\t--batch-size 32 \\\n\t--beta {beta} \\\n\t--alpha {alpha} \\\n\t--margin {margin} \\\n\t--mixup 0 \\\n\t--id 1"
            # file_name = f"train_{i}.sh"

            # with open(os.path.join(data_dir, file_name), "w+") as f:
            #     f.write(script_skeleton)

            script_skeleton = f"cd ../../../train \n\npython train_triplet_io_i1_i2_rand.py \\\n\t--gpu 0 1 \\\n\t--data-dir /mnt/personal/hutorole/mixoe/data \\\n\t--dataset {dataset} \\\n\t--epochs {epochs} \\\n\t--batch-size 32 \\\n\t--beta {beta} \\\n\t--beta2 5.0 \\\n\t--alpha {alpha} \\\n\t--margin {margin} \\\n\t--mixup 2 \\\n\t--id 1"
            file_name = f"train_{i}_mx.sh"

            with open(os.path.join(data_dir, file_name), "w+") as f:
                f.write(script_skeleton)

            i += 1


if __name__ == "__main__":
    dr = "io_i1_i2_rand"
    data_dir = "./scripts/{}/aircraft".format(dr)
    alpha = 1.0
    beta = 5.0

    main(data_dir, alpha, beta)

    data_dir = "./scripts/{}/bird".format(dr)

    main(data_dir, alpha, beta)

    data_dir = "./scripts/{}/car".format(dr)

    main(data_dir, alpha, beta)

    data_dir = "./scripts/{}/butterfly".format(dr)

    main(data_dir, alpha, beta)
