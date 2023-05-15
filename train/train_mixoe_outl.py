import torchvision.transforms as trn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import csv
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import os
import argparse
import sys

sys.path.append('..')

from models import resnet50
from eval.eval import main_eval
from utils import (
    FinegrainedDataset, SPLIT_NUM_CLASSES, INET_SPLITS, WebVision,
    AverageMeter, accuracy
)

class SoftCE(nn.Module):
    def __init__(self, reduction="mean"):
        super(SoftCE, self).__init__()
        self.reduction = reduction

    def forward(self, logits, soft_targets):
        preds = logits.log_softmax(dim=-1)
        assert preds.shape == soft_targets.shape

        loss = torch.sum(-soft_targets * preds, dim=-1)

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(
                "Reduction type '{:s}' is not supported!".format(self.reduction))


# /////////////// Setup ///////////////
# Arguments
parser = argparse.ArgumentParser(description='Trains a classifier')
# Dataset options
parser.add_argument('--dataset', type=str, choices=['bird', 'butterfly', 'car', 'aircraft'],
                    help='Choose the dataset', required=True)
parser.add_argument('--data-dir', type=str, default='../data')
parser.add_argument('--info-dir', type=str, default='../info')
parser.add_argument('--split', '-s', type=int, default=0)
# Model options
parser.add_argument('--model', '-m', type=str,
                    default='rn', help='Choose architecture.')
# MixOE options
parser.add_argument('--mix-op', type=str,
                    choices=['mixup', 'cutmix'], required=True)
parser.add_argument('--alpha', type=float, default=1.0,
                    help='Parameter for Beta distribution.')
parser.add_argument('--beta', type=float, default=5.0,
                    help='Weighting factor for the OE objective.')
parser.add_argument('--oe-set', type=str,
                    default='WebVision', choices=['WebVision'])
# Optimization options
parser.add_argument('--torch-seed', '-ts', type=int,
                    default=0, help='Random seed.')
parser.add_argument('--epochs', '-e', type=int, default=10,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='The initial learning rate.')
parser.add_argument('--batch-size', '-b', type=int,
                    default=32, help='Batch size.')
parser.add_argument('--test-bs', type=int, default=100)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float,
                    default=0.00001, help='Weight decay (L2 penalty).')
# Checkpoints
parser.add_argument('--save-dir', type=str, default=None,
                    help='Folder to save checkpoints.')
# Acceleration
parser.add_argument('--gpu', nargs='*', type=int, default=[0, 1])
parser.add_argument('--prefetch', type=int, default=2,
                    help='Pre-fetching threads.')

parser.add_argument('--outlier_num', type=int, default=0)
parser.add_argument('--outlier_classes', type=int, default=0)
args = parser.parse_args()


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
        1 + np.cos(step / total_steps * np.pi))


# /////////////// Training ///////////////
# train function
def train(oes_min, oes_max):
    net.train()  # enter train mode

    current_lr = scheduler.get_last_lr()[0]
    losses = AverageMeter('Loss', ':.4e')
    id_losses = AverageMeter('ID Loss', ':.4e')
    mixed_losses = AverageMeter('Mixed Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    batch_iter = tqdm(train_loader, total=len(train_loader),
                      desc='Batch', leave=False, position=2)

    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    oe_loader.dataset.offset = np.random.randint(len(oe_loader.dataset))
    oe_iter = iter(oe_loader)

    for bi, (x, y) in enumerate(batch_iter):
        bs = x.size(0)

        try:
            oe_x, _ = next(oe_iter)
        except StopIteration:
            continue
        assert bs == oe_x.size(0)

        x, y = x.cuda(), y.cuda()
        oe_x = oe_x.cuda()
        one_hot_y = torch.zeros(bs, num_classes).cuda()
        one_hot_y.scatter_(1, y.view(-1, 1), 1)

        # ID loss
        logits = net(x)
        id_loss = F.cross_entropy(logits, y)

        # MixOE loss
        # build mixed samples
        lam = np.random.beta(args.alpha, args.alpha)

        mixed_x = lam * x + (1 - lam) * oe_x
        mixed_x = net(mixed_x)

        oe_y = torch.ones(oe_x.size(0), num_classes).cuda() / num_classes
        soft_labels = lam * one_hot_y + (1 - lam) * oe_y

        # construct soft labels and compute loss
        mixed_loss = soft_xent(mixed_x, soft_labels)

        # Total loss
        loss = id_loss + args.beta * mixed_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        acc1, acc5 = accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item(), x.size(0))
        id_losses.update(id_loss.item(), x.size(0))
        mixed_losses.update(mixed_loss.item(), x.size(0))
        top1.update(acc1, x.size(0))
        top5.update(acc5, x.size(0))

    print_message = f'Epoch [{epoch:3d}] | ID Loss: {id_losses.avg:.4f}, Mixed Loss: {mixed_losses.avg:.4f}, Top1 Acc: {top1.avg:.2f}, Top5 Acc: {top5.avg:.2f}'
    tqdm.write(print_message)

    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/id_loss', id_losses.avg, epoch)
    writer.add_scalar('train/mixed_loss', mixed_losses.avg, epoch)
    writer.add_scalar('train/acc_top1', top1.avg, epoch)
    writer.add_scalar('train/acc_top5', top5.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)

    return oes_min, oes_max

# test function


def test():
    net.eval()

    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            output = net(x)

            acc1, acc5 = accuracy(output, y, topk=(1, 5))
            top1.update(acc1, x.size(0))
            top5.update(acc5, x.size(0))

    print_message = f'Evaluation  | Top1 Acc: {top1.avg:.2f}, Top5 Acc: {top5.avg:.2f}\n'
    tqdm.write(print_message)

    writer.add_scalar('test/acc_top1', top1.avg, epoch)
    writer.add_scalar('test/acc_top5', top5.avg, epoch)

    return top1.avg


if __name__ == "__main__":
    name_row = ["method", "alpha", "beta", "epochs",
                "batch_size", "no_outlier_classes", "no_outliers"]
    splits = 3
    runs = args.runs

    for spl in range(splits):
        for rn in range(runs):
            name_row += [f"split{spl}_run{rn}_acc",
                         f"split{spl}_run{rn}_tnr95_coarse", f"split{spl}_run{rn}_tnr95_fine"]
        name_row += [f"mean_acc_split{spl}", f"std_acc_split{spl}", f"mean_tnr95_fine_split{spl}",
                     f"std_tnr95_fine_split{spl}", f"mean_tnr95_coarse_split{spl}", f"std_tnr95_coarse_split{spl}"]
    name_row += ["mean_acc_across_splits", "std_acc_across_splits", "mean_tnr95_fine_across_splits",
                 "std_tnr95_fine_across_splits", "mean_tnr95_coarse_across_splits", "std_tnr95_coarse_across_splits"]

    last_row = [args.mix_op, args.alpha, args.beta, args.epochs,
                args.batch_size, args.outlier_classes, args.outlier_num]

    csv_file_name = f"/home/hutorole/code/mixoe/csv_results2/dataset={args.dataset}_mix={args.mix_op}_alpha={args.alpha}_beta={args.beta}_outl-cls={args.outlier_classes}_outl-num={args.outlier_num}_epochs={args.epochs}.csv"
    print("csv file: ", csv_file_name)

    all_acc, all_ftnr, all_ctnr = [], [], []
    for spl in range(splits):
        runs_acc, runs_ftnr, runs_ctnr = [], [], []
        for rn in range(runs):
            print("split, run: ", spl, rn)

            DATA_DIR = {
                'bird': os.path.join(args.data_dir, 'bird', 'images'),
                'butterfly': os.path.join(args.data_dir, 'butterfly', 'images_small'),
                'dog': os.path.join(args.data_dir, 'dog'),
                'car': os.path.join(args.data_dir, 'car'),
                'aircraft': os.path.join(args.data_dir, 'aircraft', 'images')
            }

            # Set random seed for torch
            torch.manual_seed(args.torch_seed)

            # Create ID/OOD splits
            all_classes = list(range(sum(SPLIT_NUM_CLASSES[args.dataset])))
            rng = np.random.RandomState(spl)
            id_classes = list(rng.choice(
                all_classes, SPLIT_NUM_CLASSES[args.dataset][0], replace=False))
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
                image_dir=DATA_DIR[args.dataset],
                info_filepath=os.path.join(
                    args.info_dir, args.dataset, 'train.txt'),
                class_list=id_classes,
                transform=train_transform
            )
            test_set = FinegrainedDataset(
                image_dir=DATA_DIR[args.dataset],
                info_filepath=os.path.join(
                    args.info_dir, args.dataset, 'val.txt'),
                class_list=id_classes,
                transform=test_transform
            )
            num_classes = train_set.num_classes
            assert num_classes == len(id_classes)
            print(
                f'Train samples: {len(train_set)}, Val samples: {len(test_set)}')

            if args.oe_set == 'WebVision':
                class_to_be_removed = []
                for l in INET_SPLITS.values():
                    class_to_be_removed.extend(l)
                oe_set = WebVision(
                    root=os.path.join(args.data_dir, 'WebVision'), transform=train_transform,
                    concept_list=class_to_be_removed, exclude=True,
                    num_concepts=args.outlier_classes, num_samples=args.outlier_num
                )
            else:
                raise NotImplementedError

            if args.outlier_num == 0:
                last_row[6] = len(oe_set)

            train_loader = DataLoader(
                train_set, batch_size=args.batch_size, shuffle=True,
                num_workers=args.prefetch, pin_memory=False, drop_last=True
            )
            test_loader = DataLoader(
                test_set, batch_size=args.test_bs, shuffle=False,
                num_workers=args.prefetch, pin_memory=False
            )

            # if (args.outlier_num < len(train_set) and args.outlier_num):
            if len(oe_set) < len(train_set):
                matched_oe_set = Subset(oe_set, np.random.choice(
                    len(oe_set), len(train_set), replace=True))
                oe_set = matched_oe_set

                print("Matched oe_set: ", len(oe_set))

            print("Outliers size: ", len(oe_set))

            oe_loader = DataLoader(
                oe_set, batch_size=args.batch_size, shuffle=True,
                num_workers=args.prefetch, pin_memory=False
            )

            # Set up checkpoint directory and tensorboard writer
            # if args.save_dir is None:
            mixoe_related = f'{args.mix_op}'
            mixoe_related += f'_{args.oe_set}_alpha={args.alpha:.1f}_beta={args.beta:.1f}'

            args.save_dir = os.path.join(
                '../checkpoints', args.dataset,
                f'split_{spl}',
                f'{args.model}_{mixoe_related}_epochs={args.epochs}_bs={args.batch_size}_ind={rn}_outl_class={args.outlier_classes}_outl_num={args.outlier_num}'
            )

            # else:
            #     assert 'checkpoints' in args.save_dir, \
            #         "If 'checkpoints' not in save_dir, then you may have an unexpected directory for writer..."
            chkpnt_path = os.path.join(
                args.save_dir, f'seed_{args.torch_seed}.pth')

            print("Checkpoint path: ", chkpnt_path)

            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            elif os.path.isfile(chkpnt_path):
                print('*********************************')
                print('* The checkpoint already exists *')
                print('*********************************')

            writer = SummaryWriter(
                args.save_dir.replace('checkpoints', 'runs'))

            # Set up GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
                map(lambda x: str(x), args.gpu))

            # Create model
            if args.model == 'rn':
                # net = torch.load(
                #     "/home/hutorole/code/mixoe/models/resnet50.pth", map_location='cuda')
                net = resnet50()
                pretrained_model_file = f'../checkpoints/{args.dataset}/split_{spl}/rn_baseline_epochs=90_bs=32/seed_0.pth'
                state_dict = torch.load(pretrained_model_file)
            else:
                raise NotImplementedError

            in_features = net.fc.in_features
            new_fc = nn.Linear(in_features, num_classes)
            net.fc = new_fc

            try:
                net.load_state_dict(state_dict)
            except RuntimeError:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.` caused by nn.DataParallel
                    new_state_dict[name] = v
                net.load_state_dict(new_state_dict)

            net.cuda()
            if torch.cuda.device_count() > 1:
                net = torch.nn.DataParallel(net)
            cudnn.benchmark = True  # fire on all cylinders

            # Optimizer and scheduler
            optimizer = optim.SGD(
                net.parameters(), args.lr, momentum=args.momentum,
                weight_decay=args.decay, nesterov=True
            )

            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: cosine_annealing(
                    step,
                    args.epochs * len(train_loader),
                    1,  # since lr_lambda computes multiplicative factor
                    1e-6 / args.lr)
            )

            soft_xent = SoftCE()

            # Main loop
            epoch_iter = tqdm(list(range(1, args.epochs+1)), total=args.epochs, desc='Epoch',
                              leave=True, position=1)


            best_acc1 = 0
            for epoch in epoch_iter:
                
                oes_min, oes_max = train(oes_min, oes_max)
                acc1 = test()

                if acc1 > best_acc1:
                    # Save model
                    torch.save(
                        net.state_dict(),
                        chkpnt_path
                    )
                best_acc1 = max(acc1, best_acc1)

            acc, tnrs = main_eval(chkpnt_path)

            # print(acc, tnrs)

            last_row.append(acc)
            last_row.append(tnrs['coarse'])
            last_row.append(tnrs['fine'])

            runs_acc.append(acc)
            runs_ctnr.append(tnrs['coarse'])
            runs_ftnr.append(tnrs['fine'])

        last_row.append(np.mean(runs_acc))
        last_row.append(np.std(runs_acc))
        last_row.append(np.mean(runs_ctnr))
        last_row.append(np.std(runs_ctnr))
        last_row.append(np.mean(runs_ftnr))
        last_row.append(np.std(runs_ftnr))

        all_acc.extend(runs_acc)
        all_ctnr.extend(runs_ctnr)
        all_ftnr.extend(runs_ftnr)

    last_row.append(np.mean(all_acc))
    last_row.append(np.std(all_acc))
    last_row.append(np.mean(all_ctnr))
    last_row.append(np.std(all_ctnr))
    last_row.append(np.mean(all_ftnr))
    last_row.append(np.std(all_ftnr))

    assert len(name_row) == len(last_row)

    with open(csv_file_name, 'w+') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow(name_row)
        csv_writer.writerow(last_row)
