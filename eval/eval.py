# Modified from MixOE to collect and return results

from models import resnet50
from utils import (
    silence_PIL_warnings, FinegrainedDataset,
    print_measures_with_std, print_measures, SPLIT_NUM_CLASSES, INET_SPLITS, WebVision
)
import utils.calculate_log as callog
import torchvision.transforms as trn
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import torch.nn as nn
import torch
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import os
import datetime
import sys
import random
import faiss
sys.path.append('..')


# EXIF warning silent
silence_PIL_warnings()


def concat(x): return np.concatenate(x, axis=0)
def to_np(x): return x.data.cpu().numpy()


def get_baseline_scores(model, loader, in_dist=False):
    _score = []
    total = 0
    correct = []

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(loader), total=len(loader)):
            data = data.cuda()
            output = model(data)

            if in_dist:
                correct.append(output.argmax(1).cpu().eq(target))

            smax = to_np(F.softmax(output, dim=1))
            _score.append(np.max(smax, axis=1))

            total += data.size(0)

    if in_dist:
        return concat(_score).copy(), torch.cat(correct).numpy()  # acc/total
    else:
        return concat(_score).copy()


def get_and_print_results(in_score, model, ood_loader, ood_fn, ood, val, num_run=10, **kwargs):
    out_score = ood_fn(model, ood_loader, **kwargs)
    sample_size = min(len(in_score), len(out_score))

    aurocs, auprs, tnrs = [], [], []
    random.seed(0)
    for i in range(num_run):
        metric_results = callog.metric(
            np.array(random.sample(list(in_score), sample_size)),
            np.array(random.sample(list(out_score), sample_size))
        )
        # update stat keepers for auroc, tnr, aupr
        aurocs.append(metric_results['TMP']['AUROC'])
        tnrs.append(metric_results['TMP']['TNR'])
        auprs.append(metric_results['TMP']['AUIN'])

    auroc = np.mean(aurocs)
    aupr = np.mean(auprs)
    tnr = np.mean(tnrs)

    if num_run >= 1:
        msg = print_measures_with_std(aurocs, auprs, tnrs)
    else:
        msg = print_measures(auroc, aupr, tnr)

    return auroc, aupr, tnr, msg


def print_and_write(msg, fp):
    if isinstance(msg, str):
        print(msg)
        if fp is not None:
            fp.write(msg+'\n')
    elif isinstance(msg, list):
        print('\n'.join(msg))
        if fp is not None:
            fp.write('\n'.join(msg))
        if fp is not None:
            fp.write('\n')
    else:
        raise TypeError


# adapted from https://blog.munhou.com/2022/12/01/Detecting%20Out-of-Distribution%20Samples%20with%20Knn/
# distance-based TNR95@TPR, not used in the final thesis
def get_normalized_trn_embeddings(net, train_set, id_set, ood_sets, k, window=False):
    train_loader = DataLoader(
        train_set, batch_size=100, shuffle=True,
        num_workers=4, pin_memory=False, drop_last=True
    )
    id_loader = DataLoader(
        id_set, batch_size=100, shuffle=False,
        num_workers=4
    )
    ood_loaders = {
        k: DataLoader(
            v, batch_size=100, shuffle=True,
            num_workers=4
        ) for k, v in ood_sets.items()
    }


    with torch.no_grad():
        train_embeddings = []
        for x, y in train_loader:
            x = x.cuda()
            logits, emb = net(x, return_embeddings=True)
            train_embeddings.extend(emb.data.cpu().numpy())

        train_embeddings = np.array(train_embeddings)
        train_embeddings = (train_embeddings.T / np.linalg.norm(train_embeddings, 2, -1)).T
        print(train_embeddings.shape)

        index = faiss.IndexFlatL2(train_embeddings.shape[1])
        index.add(train_embeddings)

        id_test_embeddings = []
        for x, y in id_loader:
            x = x.cuda()
            logits, emb = net(x, return_embeddings=True)
            id_test_embeddings.extend(emb.data.cpu().numpy())

        id_test_embeddings = np.array(id_test_embeddings)
        id_test_embeddings = (id_test_embeddings.T / np.linalg.norm(id_test_embeddings, 2, -1)).T

        id_test_distances, id_test_k_closest = index.search(id_test_embeddings, k)
        id_test_scores = id_test_distances[:, -1]
        id_test_scores.sort()
        threshold = id_test_scores[round(0.95 * len(id_test_scores))]
        print("Threshold: ", threshold)

        ood_test_embeddings_fine = []
        for x, y in ood_loaders['fine']:
            x = x.cuda()
            logits, emb = net(x, return_embeddings=True)
            ood_test_embeddings_fine.extend(emb.data.cpu().numpy())

        ood_test_embeddings_fine = np.array(ood_test_embeddings_fine)
        ood_test_embeddings_fine = (ood_test_embeddings_fine.T / np.linalg.norm(ood_test_embeddings_fine, 2, -1)).T

        if not window:
            ood_test_distances_fine, id_test_k_closest = index.search(ood_test_embeddings_fine, k)
            ood_test_scores_fine = ood_test_distances_fine[:, -1]
        else:
            ood_test_distances_fine, id_test_k_closest = index.search(ood_test_embeddings_fine, k+window)
            ood_test_scores_fine = np.mean(ood_test_distances_fine[:, -2*window:], axis=1)

        ood_test_embeddings_coarse = []
        for x, y in ood_loaders['coarse']:
            x = x.cuda()
            logits, emb = net(x, return_embeddings=True)
            ood_test_embeddings_coarse.extend(emb.data.cpu().numpy())

        ood_test_embeddings_coarse = np.array(ood_test_embeddings_coarse)
        ood_test_embeddings_coarse = (ood_test_embeddings_coarse.T / np.linalg.norm(ood_test_embeddings_coarse, 2, -1)).T

        if not window:
            ood_test_distances_coarse, id_test_k_closest = index.search(ood_test_embeddings_coarse, k)
            ood_test_scores_coarse = ood_test_distances_coarse[:, -1]
        else:
            ood_test_distances_coarse, id_test_k_closest = index.search(ood_test_embeddings_coarse, k+window)
            ood_test_scores_coarse = np.mean(ood_test_distances_coarse[:, -2*window:], axis=1)

        tp, fp_fine, fp_coarse, tn_fine, tn_coarse, fn = 0, 0, 0, 0, 0, 0

        tp += np.sum(id_test_scores <= threshold)
        fn += np.sum(id_test_scores > threshold)

        tn_fine += np.sum(ood_test_scores_fine > threshold)
        fp_fine += np.sum(ood_test_scores_fine <= threshold)

        tn_coarse += np.sum(ood_test_scores_coarse > threshold)
        fp_coarse += np.sum(ood_test_scores_coarse <= threshold)

        tnr_fine = tn_fine / (tn_fine + fp_fine)
        tnr_coarse = tn_coarse / (tn_coarse + fp_coarse)
        tpr = tp / (tp + fn)

        return tpr, tnr_fine, tnr_coarse


def main_eval(model_file, train_set, k_near, window=False):

    model_file = model_file
    data_dir = '/mnt/personal/hutorole/mixoe/data'
    info_dir = '../info'
    batch_size = 100
    ood = 'msp'
    temp = 1
    val = False
    save_to_file = False
    overwrite = False
    gpu = [0, 1]
    prefetch = 4

    DATA_DIR = {
        'bird': os.path.join(data_dir, 'bird', 'images'),
        'butterfly': os.path.join(data_dir, 'butterfly', 'images_small'),
        'car': os.path.join(data_dir, 'car'),
        'aircraft': os.path.join(data_dir, 'aircraft', 'images')
    }
    assert os.path.isfile(model_file)

    temp = model_file.split('/')
    dataset = temp[temp.index('checkpoints')+1]
    split_idx = int(temp[temp.index('checkpoints')+2].split('_')[-1])
    arch = model_file.split('/')[-2].split('_')[0]

    print(f'Dataset: {dataset}, Split ID: {split_idx}, Arch: {arch}')

    # Set up txt file for recording results
    if save_to_file:
        result_root = '/'.join(model_file.replace('checkpoints',
                               'results').split('/')[:-1])
        if not os.path.exists(result_root):
            os.makedirs(result_root)

        prefix = model_file.split('/')[-1].split('.')[0]
        if val:
            prefix = 'val_' + prefix

        filename = '%s_%s.txt' % (prefix, ood)

        result_filepath = os.path.join(result_root, filename)
        if os.path.isfile(result_filepath) and not overwrite:
            f = open(result_filepath, 'a')
        else:
            f = open(result_filepath, 'w')
        f.write('\n\nTime: %s\n' % str(datetime.datetime.now()))
    else:
        f = None

    # Create ID/OOD splits
    all_classes = list(range(sum(SPLIT_NUM_CLASSES[dataset])))
    rng = np.random.RandomState(split_idx)
    id_classes = list(rng.choice(
        all_classes, SPLIT_NUM_CLASSES[dataset][0], replace=False))
    ood_classes = [c for c in all_classes if c not in id_classes]

    # Datasets and dataloaders
    # In-distribution
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    test_transform = trn.Compose([
        trn.Resize((512, 512)),
        trn.CenterCrop(448),
        trn.ToTensor(),
        trn.Normalize(mean, std),
    ])

    if val:
        id_set = FinegrainedDataset(
            image_dir=DATA_DIR[dataset],
            info_filepath=os.path.join(info_dir, dataset, 'val.txt'),
            class_list=id_classes,
            transform=test_transform
        )
    else:
        id_set = FinegrainedDataset(
            image_dir=DATA_DIR[dataset],
            info_filepath=os.path.join(info_dir, dataset, 'test.txt'),
            class_list=id_classes,
            transform=test_transform
        )

    id_loader = DataLoader(
        id_set, batch_size=batch_size, shuffle=False,
        num_workers=prefetch
    )
    num_classes = id_set.num_classes
    assert num_classes == len(id_classes)

    # Out-of-distribution
    if val:
        class_to_be_removed = []
        for l in INET_SPLITS.values():
            class_to_be_removed.extend(l)

        ood_sets = {}
        temp = WebVision(
            root='/mnt/personal/hutorole/mixoe/data/WebVision', transform=test_transform,
            concept_list=class_to_be_removed, exclude=True
        )
        random.seed(0)
        ood_sets['WebVision'] = Subset(
            temp, random.sample(range(len(temp)), len(id_set)))
    else:
        ood_sets = {}

        others = [k for k in DATA_DIR.keys() if k != dataset]
        # combine other datasets into a single dataset
        for i, dset_name in enumerate(others):
            if i == 0:
                others_set = FinegrainedDataset(
                    image_dir=DATA_DIR[dset_name],
                    info_filepath=os.path.join(
                        info_dir, dset_name, 'test.txt'),
                    transform=test_transform
                )
            else:
                temp = FinegrainedDataset(
                    image_dir=DATA_DIR[dset_name],
                    info_filepath=os.path.join(
                        info_dir, dset_name, 'test.txt'),
                    transform=test_transform
                )
                others_set.samples.extend(temp.samples)
        random.seed(0)
        others_set.samples = random.sample(others_set.samples, len(id_set))
        ood_sets['coarse'] = others_set

        # fine-grained hold-out set
        holdout_set = FinegrainedDataset(
            image_dir=DATA_DIR[dataset],
            info_filepath=os.path.join(info_dir, dataset, 'test.txt'),
            class_list=ood_classes,
            transform=test_transform
        )
        ood_sets['fine'] = holdout_set

    ood_loaders = {
        k: DataLoader(
            v, batch_size=batch_size, shuffle=True,
            num_workers=prefetch
        ) for k, v in ood_sets.items()
    }

    # Set up GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        map(lambda x: str(x), gpu))

    # Get model
    if arch == 'rn':
        net = resnet50()
    else:
        raise NotImplementedError

    in_features = net.fc.in_features
    new_fc = nn.Linear(in_features, num_classes)
    net.fc = new_fc
    state_dict = torch.load(model_file)
    try:
        net.load_state_dict(state_dict)
    except RuntimeError:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.` caused by nn.DataParallel
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)

    net.eval()
    net.cuda()
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    ood_fn = get_baseline_scores

    # Get ID scores
    in_score, in_correct = ood_fn(net, id_loader, in_dist=True)
    acc = np.mean(in_correct)
    print_and_write([
        'Acc:         %.2f' % (100.*acc)
    ], f)

    tnrs = {}
    # Get OOD scores
    for ood_name, ood_loader in ood_loaders.items():
        auroc, aupr, tnr, msg = get_and_print_results(
            in_score, net, ood_loader,
            ood_fn, ood, val
        )

        print_and_write(['\n', ood_name], f)
        print_and_write(msg, f)
        tnrs[ood_name] = tnr

    if f is not None:
        f.write('='*50)
        f.close()

    tpr, tnr_fine, tnr_coarse = get_normalized_trn_embeddings(net, train_set, id_set, ood_sets, k_near, window)

    msg = 'Detection through thresholding:\nTPR:\t\t\t{:.2f} TNR_fine:\t\t\t{:.2f} TNR_coarse:\t\t\t{:2f}'.format(100 * tpr, 100 * tnr_fine, 100 * tnr_coarse)
    print_and_write(msg, f)

    return acc, tnrs, tpr, tnr_fine, tnr_coarse
