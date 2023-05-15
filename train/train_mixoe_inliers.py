import wandb
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
import os
import argparse
import sys

sys.path.append('..')
from utils import (
    AverageMeter, accuracy, data_load
)

from eval.eval import main_eval


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
parser.add_argument('--alpha', type=float, default=1.0,
                    help='Parameter for Beta distribution.')
parser.add_argument('--beta', type=float, default=1.0,
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
parser.add_argument('--prefetch', type=int, default=4,
                    help='Pre-fetching threads.')
parser.add_argument('--run', type=int, default=0, help='Run number')
args = parser.parse_args()


wandb.init(
    # set the wandb project where this run will be logged
    # project="MixOE_{}_10".format(args.dataset),
    project="{}".format(args.dataset),

    # track hyperparameters and run metadata
    config={
        "learning_rate": args.lr,
        "architecture": args.model,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "mix_op": "mixoe_inliers",
    },

    # group="split{}".format(args.split)
    group="mixoe_inliers_epochs={}".format(args.epochs),

    name="split={}_run={}".format(args.split, args.run)
)


train_set, test_set, oe_set = data_load.load_data(
    args.data_dir, args.dataset, args.info_dir, args.split)

num_classes = train_set.num_classes

train_loader = DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=False, drop_last=True
)
test_loader = DataLoader(
    test_set, batch_size=args.test_bs, shuffle=False,
    num_workers=args.prefetch, pin_memory=False
)
oe_loader = DataLoader(
    oe_set, batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=False
)

# Set up checkpoint directory and tensorboard writer
if args.save_dir is None:
    mixoe_related = 'mixoe_inliers'
    mixoe_related += f'_{args.oe_set}_alpha={args.alpha:.1f}_beta={args.beta:.1f}_run={args.run}'

    args.save_dir = os.path.join(
        '../checkpoints', args.dataset,
        f'split_{args.split}',
        f'{args.model}_{mixoe_related}_epochs={args.epochs}_bs={args.batch_size}'
    )
else:
    assert 'checkpoints' in args.save_dir, \
        "If 'checkpoints' not in save_dir, then you may have an unexpected directory for writer..."
chkpnt_path = os.path.join(args.save_dir, f'seed_{args.torch_seed}.pth')

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
elif os.path.isfile(chkpnt_path):
    print('*********************************')
    print('* The checkpoint already exists *')
    print('*********************************')

writer = SummaryWriter(args.save_dir.replace('checkpoints', 'runs'))

# Set up GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(lambda x: str(x), args.gpu))

net = data_load.load_net(args.dataset, args.split, num_classes)

# Optimizer and scheduler
optimizer = optim.SGD(
    net.parameters(), args.lr, momentum=args.momentum,
    weight_decay=args.decay, nesterov=True
)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
        1 + np.cos(step / total_steps * np.pi))


scheduler = optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.lr)
)

soft_xent = SoftCE()

# /////////////// Training ///////////////
# train function


def train():
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

    for x, y in batch_iter:
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

        # Mixup with inliers
        lam = np.random.beta(args.alpha, args.alpha)
        indices = torch.randperm(x.shape[0])
        mixed_inl_x = lam * x + (1 - lam) * x[indices]
        soft_labels = lam * one_hot_y + (1 - lam) * one_hot_y[indices]
        mixed_inl_loss = soft_xent(net(mixed_inl_x), soft_labels)

        # Total loss
        loss = id_loss + args.beta * mixed_inl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        acc1, acc5 = accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item(), x.size(0))
        id_losses.update(id_loss.item(), x.size(0))
        mixed_losses.update(mixed_inl_loss.item(), x.size(0))
        top1.update(acc1, x.size(0))
        top5.update(acc5, x.size(0))

    wandb.log({"acc1": top1.avg, "acc5": top5.avg, "id_loss": id_losses.avg,
                "mixed_inl_loss": mixed_losses.avg, "loss": losses.avg})

    print_message = f'Epoch [{epoch:3d}] | ID Loss: {id_losses.avg:.4f}, Mixed Loss: {mixed_losses.avg:.4f}, ' \
        f'Top1 Acc: {top1.avg:.2f}, Top5 Acc: {top5.avg:.2f}'
    tqdm.write(print_message)

    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/id_loss', id_losses.avg, epoch)
    writer.add_scalar('train/mixed_loss', mixed_losses.avg, epoch)
    writer.add_scalar('train/acc_top1', top1.avg, epoch)
    writer.add_scalar('train/acc_top5', top5.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)

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


# Main loop
epoch_iter = tqdm(list(range(1, args.epochs+1)), total=args.epochs, desc='Epoch',
                  leave=True, position=1)

best_acc1 = 0
for epoch in epoch_iter:
    train()
    acc1 = test()

    if acc1 > best_acc1:
        # Save model
        torch.save(
            net.state_dict(),
            chkpnt_path
        )
    best_acc1 = max(acc1, best_acc1)


acc, tnrs, tpr, tnr_fine, tnr_coarse = main_eval(chkpnt_path, train_set, 20)
wandb.log({"test_acc": acc, "tnr_fine_conf": tnrs['fine'], "tnr_coarse_conf": tnrs['coarse'],
           "tpr": tpr, "tnr_fine_dist": tnr_fine, "tnr_coarse_dist": tnr_coarse})

wandb.finish()
