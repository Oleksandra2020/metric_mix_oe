import torchvision.transforms as trn
import torch
import torch.nn as nn
import numpy as np
import os
from collections import OrderedDict
import torch.backends.cudnn as cudnn

from utils import (
    FinegrainedDataset, SPLIT_NUM_CLASSES, INET_SPLITS, WebVision,
)
# from torchvision.models import resnet50
from models import resnet50, resnet50_m


def load_data(data_dir, dataset, info_dir, spl, num_outliers=0, num_classes=0):
    DATA_DIR = {
        'bird': os.path.join(data_dir, 'bird', 'images'),
        'butterfly': os.path.join(data_dir, 'butterfly', 'images_small'),
        'dog': os.path.join(data_dir, 'dog'),
        'car': os.path.join(data_dir, 'car'),
        'aircraft': os.path.join(data_dir, 'aircraft', 'images')
    }

    # Create ID/OOD splits
    all_classes = list(range(sum(SPLIT_NUM_CLASSES[dataset])))
    rng = np.random.RandomState(spl)
    id_classes = list(rng.choice(
        all_classes, SPLIT_NUM_CLASSES[dataset][0], replace=False))
    ood_classes = [c for c in all_classes if c not in id_classes]
    print(
        f'# ID classes: {len(id_classes)}, # OOD classes: {len(ood_classes)}')

    # Datasets and dataloaders
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = trn.Compose([
        trn.Resize((512, 512)),
        trn.RandomCrop((448, 448)),
        trn.RandomHorizontalFlip(),
        trn.ColorJitter(brightness=32./255., saturation=0.5),
        trn.ToTensor(),
        trn.Normalize(mean, std),
    ])
    test_transform = trn.Compose([
        trn.Resize((512, 512)),
        trn.CenterCrop(448),
        trn.ToTensor(),
        trn.Normalize(mean, std),
    ])

    train_set = FinegrainedDataset(
        image_dir=DATA_DIR[dataset],
        info_filepath=os.path.join(
            info_dir, dataset, 'train.txt'),
        class_list=id_classes,
        transform=train_transform
    )
    test_set = FinegrainedDataset(
        image_dir=DATA_DIR[dataset],
        info_filepath=os.path.join(
            info_dir, dataset, 'val.txt'),
        class_list=id_classes,
        transform=test_transform
    )
    assert train_set.num_classes == len(id_classes)
    print(
        f'Train samples: {len(train_set)}, Val samples: {len(test_set)}')

    class_to_be_removed = []
    for l in INET_SPLITS.values():
        class_to_be_removed.extend(l)

    oe_set = WebVision(
        root=os.path.join(data_dir, 'WebVision'), transform=train_transform,
        concept_list=class_to_be_removed, exclude=True, num_concepts=num_classes, num_samples=num_outliers
    )
    return train_set, test_set, oe_set


def load_net(dataset, split, num_classes, manifold=False):
    if manifold:
        net = resnet50_m()
    else:
        net = resnet50()
    pretrained_model_file = f'/home/hutorole/code/metric_oe/checkpoints/{dataset}/split_{split}/rn_baseline_epochs=90_bs=32_run=0/seed_0.pth'
    state_dict = torch.load(pretrained_model_file)

    in_features = net.fc.in_features
    new_fc = nn.Linear(in_features, num_classes)
    net.fc = new_fc

    try:
        net.load_state_dict(state_dict)
    except RuntimeError:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.` caused by nn.DataParallel
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)

    net.cuda()
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    cudnn.benchmark = True  # fire on all cylinders

    return net