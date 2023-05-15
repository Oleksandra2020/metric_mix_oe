import os


def main(data_dir, alpha, beta):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    dataset = data_dir.split('/')[-1]

    if dataset == 'air':
        dataset = 'aircraft'

    epochs = 10

    i = 0
    splits = 3
    runs = 5
    for s in range(splits):

        for r in range(runs):

            # script_skeleton = f"cd ../../../train \n\npython train_baseline.py \\\n\t--gpu 0 1 \\\n\t--data-dir /mnt/personal/hutorole/mixoe/data \\\n\t--dataset {dataset} \\\n\t--epochs {epochs} \\\n\t--batch-size 32 \\\n\t--split {s} \\\n\t--run {r}"
            # file_name = f"train_baseline_{i}.sh"

            # script_skeleton = f"cd ../../../train \n\npython train_mixoe.py \\\n\t--gpu 0 1 \\\n\t--data-dir /mnt/personal/hutorole/mixoe/data \\\n\t--dataset {dataset} \\\n\t--epochs {epochs} \\\n\t--batch-size 32 \\\n\t--mix-op mixup \\\n\t--split {s} \\\n\t--run {r} \\\n\t--alpha {alpha} \\\n\t--beta {beta}"
            # file_name = f"train_mixoe_{i}.sh"

            script_skeleton = f"cd ../../../train \n\npython train_align.py \\\n\t--gpu 0 1 \\\n\t--data-dir /mnt/personal/hutorole/mixoe/data \\\n\t--dataset {dataset} \\\n\t--epochs {epochs} \\\n\t--batch-size 32 \\\n\t--split {s} \\\n\t--run {r} \\\n\t--alpha {alpha} \\\n\t--beta {beta}"
            file_name = f"train_{i}.sh"
    
            # script_skeleton = f"cd ../../../train \n\npython train_knn.py \\\n\t--gpu 0 1 \\\n\t--data-dir /mnt/personal/hutorole/mixoe/data \\\n\t--dataset {dataset} \\\n\t--epochs {epochs} \\\n\t--batch-size 32 \\\n\t--split {s} \\\n\t--run {r} \\\n\t--alpha {alpha} \\\n\t--beta {beta}"
            # file_name = f"train_{i}.sh"

            # script_skeleton = f"cd ../../../train \n\npython train_mixoe_manifold.py \\\n\t--gpu 0 1 \\\n\t--data-dir /mnt/personal/hutorole/mixoe/data \\\n\t--dataset {dataset} \\\n\t--epochs {epochs} \\\n\t--batch-size 32 \\\n\t--split {s} \\\n\t--run {r} \\\n\t--alpha {alpha} \\\n\t--beta {beta}"
            # file_name = f"train_{i}.sh"

            with open(os.path.join(data_dir, file_name), "w+") as f:
                f.write(script_skeleton)

            i += 1


if __name__ == "__main__":
    dr = "align"
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
