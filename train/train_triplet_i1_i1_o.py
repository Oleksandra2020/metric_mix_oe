# Modified from MixOE
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
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

from utils import (
    AverageMeter, accuracy, data_load
)
from models import resnet50
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


class L2_norm(nn.Module):
    def __init__(self):
        super(L2_norm, self).__init__()

    def forward(self, x):
        return F.normalize(x, p=2, dim=1)


# /////////////// Setup ///////////////
# Arguments
parser = argparse.ArgumentParser(description='Trains a classifier')
# Dataset options
parser.add_argument('--dataset', type=str, choices=['bird', 'butterfly', 'car', 'aircraft'],
                    help='Choose the dataset', required=True)
parser.add_argument('--data-dir', type=str, default='../data')
parser.add_argument('--info-dir', type=str, default='../info')
# Model options
parser.add_argument('--model', '-m', type=str,
                    default='rn', help='Choose architecture.')
parser.add_argument('--beta2', type=float, default=5.0,
                    help='Weighting factor for the triplet norm.')
parser.add_argument('--beta', type=float, default=0.1,
                    help='Weighting factor for the triplet loss.')
parser.add_argument('--alpha', type=float, default=0.0,
                    help='Beta distribution parameter')
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

parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--margin', type=float, default=0.05)
parser.add_argument('--mixup', type=int, default=0)
parser.add_argument('--id', type=int, default=1)
args = parser.parse_args()


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
        1 + np.cos(step / total_steps * np.pi))


# /////////////// Training ///////////////
# train function
def train():
    net.train()  # enter train mode

    current_lr = scheduler.get_last_lr()[0]
    losses = AverageMeter('Loss', ':.4e')
    id_losses = AverageMeter('ID Loss', ':.4e')
    triplet_losses = AverageMeter('Triplet Loss', ':.4e')
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
        logits, embeddings = net(x, return_embeddings=True)
        if args.id:
            id_loss = F.cross_entropy(logits, y)

        # Mixup loss
        lam = np.random.beta(args.alpha, args.alpha)
        if args.mixup:
            mixed_x = lam * x + (1 - lam) * oe_x
            mixed_x = net(mixed_x)
            oe_y = torch.ones(oe_x.size(0), num_classes).cuda() / num_classes
            soft_labels = lam * one_hot_y + (1 - lam) * oe_y

            mixed_loss = soft_xent(mixed_x, soft_labels)

        embeddings_n = F.normalize(embeddings, p=2, dim=-1)
        oe_logits, oe_embeddings = net(oe_x, return_embeddings=True)
        oe_embeddings_n = F.normalize(oe_embeddings, p=2, dim=-1)
        distance = 1 - torch.mm(embeddings_n, oe_embeddings_n.t())

        triplet_loss = torch.zeros(1, requires_grad=True).cuda()
        count = 0
        
        for i in range(embeddings.shape[0]):
            row = distance[i]
            positive_inds = torch.where(y == y[i])[0]
            negative_inds = torch.where(y != -1)[0]

            if positive_inds.shape[0] == 1 or negative_inds.shape[0] == 1:
                continue

            count += positive_inds.shape[0]

            anchor = embeddings[i]
            negative = oe_embeddings[torch.argmin(row[negative_inds])]
            
            positive = embeddings[positive_inds]
            anchor = anchor[None, :].repeat(positive.shape[0], 1)
            negative = negative[None, :].repeat(positive.shape[0], 1)

            triplet_loss += triplet_loss_m(anchor, positive, negative)
                
        loss = args.beta * triplet_loss
        if args.id:
            loss += id_loss
        if args.mixup == 1:
            loss += mixed_loss
        if args.mixup == 2:
            loss += args.beta2 * mixed_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        acc1, acc5 = accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item(), x.size(0))
        if args.id:
            id_losses.update(id_loss.item(), x.size(0))
        if args.mixup:
            mixed_losses.update(mixed_loss.item(), x.size(0))
        if count != 0:
            triplet_losses.update(triplet_loss.item(), count)
        top1.update(acc1, x.size(0))
        top5.update(acc5, x.size(0))

    if args.mixup:
        mixed_losses_avg = mixed_losses.avg
    else:
        mixed_losses_avg = 0

    print_message = "Epoch [{:3d}] | ID Loss: {:.4f}, Triplet Loss: {:.4f}, Mixed Loss: {:.4f}, Top1 Acc: {:.2f}, Top5 Acc: {:.2f}".format(
        epoch, id_losses.avg, triplet_losses.avg, mixed_losses_avg, top1.avg, top5.avg)
    tqdm.write(print_message)

    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/id_loss', id_losses.avg, epoch)
    writer.add_scalar('train/triplet_loss', triplet_losses.avg, epoch)
    writer.add_scalar('train/margin_loss', mixed_losses_avg, epoch)
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

    print_message = 'Evaluation  | Top1 Acc: {:.2f}, Top5 Acc: {:.2f}\n'.format(
        top1.avg, top5.avg)
    tqdm.write(print_message)

    writer.add_scalar('test/acc_top1', top1.avg, epoch)
    writer.add_scalar('test/acc_top5', top5.avg, epoch)

    return top1.avg


if __name__ == "__main__":
    name_row = ["method", "alpha", "beta", "epochs",
                "batch_size", "margin", "mixup"]
    splits = 3
    runs = args.runs

    # Prepare csv file format
    for spl in range(splits):
        for rn in range(runs):
            name_row += ["split{}_run{}_acc".format(spl, rn), "split{}_run{}_tpr".format(spl, rn),
                         "split{}_run{}_tnr_fine".format(spl, rn), "split{}_run{}_tnr_coarse".format(spl, rn),
                         "split{}_run{}_tnr95_coarse".format(spl, rn), "split{}_run{}_tnr95_fine".format(spl, rn)]
        name_row += ["mean_acc_split{}".format(spl), "std_acc_split{}".format(spl), 
                     "mean_tpr_split{}".format(spl), "std_tpr_split{}".format(spl),
                     "mean_tnr_fine_split{}".format(spl), "std_tnr_fine_split{}".format(spl),
                     "mean_tnr_coarse_split{}".format(spl), "std_tnr_coarse_split{}".format(spl),
                     "mean_tnr95_fine_split{}".format(spl),
                     "std_tnr95_fine_split{}".format(spl), "mean_tnr95_coarse_split{}".format(spl), "std_tnr95_coarse_split{}".format(spl)]
    name_row += ["mean_acc_across_splits", "std_acc_across_splits",
                 "mean_tpr_across_splits", "std_tpr_across_splits",
                 "mean_tnr_fine_across_splits", "std_tnr_fine_across_splits",
                 "mean_tnr_coarse_across_splits", "std_tnr_coarse_across_splits",
                 "mean_tnr95_fine_across_splits", "std_tnr95_fine_across_splits",
                 "mean_tnr95_coarse_across_splits", "std_tnr95_coarse_across_splits"]

    last_row = ["i1_i1_o", args.alpha, args.beta, args.epochs,
                args.batch_size, args.margin, args.mixup]

    csv_file_name = "/home/hutorole/code/metric_oe/csv_results/dataset={}_beta={}_epochs={}_margin={}_mixup={}_id={}_i1_i1_o.csv".format(
        args.dataset, args.beta, args.epochs, args.margin, args.mixup, args.id)

    print("csv file: ", csv_file_name)

    all_acc, all_ftnr, all_ctnr = [], [], []
    all_tpr, all_tnr_fine, all_tnr_coarse = [], [], []
    for spl in range(splits):
        runs_acc, runs_ftnr, runs_ctnr = [], [], []
        runs_tpr, runs_tnr_fine, runs_tnr_coarse = [], [], []
        for rn in range(runs):
            print("split, run: ", spl, rn)
            # Set random seed for torch
            torch.manual_seed(args.torch_seed)

            # Prepare data
            train_set, test_set, oe_set = data_load.load_data(
                args.data_dir, args.dataset, args.info_dir, spl)
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

            mixoe_related = ''
            mixoe_related += '{}_alpha={:.1f}_beta={:.1f}'.format(
                args.oe_set, 0, args.beta)

            args.save_dir = os.path.join(
                '/home/hutorole/code/metric_oe/checkpoints', args.dataset,
                'split_{}'.format(spl),
                '{}_{}_epochs={}_bs={}_ind={}_outl_class={}_outl_num={}_margin={}_mixup={}_id={}_i1_i1_o'.format(
                    args.model, mixoe_related, args.epochs, args.batch_size, rn, 0, 0, args.margin, args.mixup, args.id)
            )

            chkpnt_path = os.path.join(
                args.save_dir, 'seed_{}.pth'.format(args.torch_seed))

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
                net = resnet50()
                pretrained_model_file = '/home/hutorole/code/mixoe/checkpoints/{}/split_{}/rn_baseline_epochs=90_bs=32/seed_0.pth'.format(
                    args.dataset, spl)
                state_dict = torch.load(pretrained_model_file)
                # if rn == 0:
                #     main_eval(pretrained_model_file, train_set, 20)
            else:
                raise NotImplementedError

            num_classes = train_set.num_classes

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
            triplet_loss_m = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=args.margin)

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

            acc, tnrs, tpr, tnr_fine, tnr_coarse = main_eval(chkpnt_path, train_set,20 )

            print(acc, tnrs)

            last_row.append(acc)
            last_row.append(tpr)
            last_row.append(tnr_fine)
            last_row.append(tnr_coarse)
            last_row.append(tnrs['coarse'])
            last_row.append(tnrs['fine'])

            runs_acc.append(acc)
            runs_tpr.append(tpr)
            runs_tnr_fine.append(tnr_fine)
            runs_tnr_coarse.append(tnr_coarse)
            runs_ctnr.append(tnrs['coarse'])
            runs_ftnr.append(tnrs['fine'])

        last_row.append(np.mean(runs_acc))
        last_row.append(np.std(runs_acc))
        last_row.append(np.mean(runs_tpr))
        last_row.append(np.std(runs_tpr))
        last_row.append(np.mean(runs_tnr_fine))
        last_row.append(np.std(runs_tnr_fine))
        last_row.append(np.mean(runs_tnr_coarse))
        last_row.append(np.std(runs_tnr_coarse))
        last_row.append(np.mean(runs_ctnr))
        last_row.append(np.std(runs_ctnr))
        last_row.append(np.mean(runs_ftnr))
        last_row.append(np.std(runs_ftnr))

        all_acc.extend(runs_acc)
        all_tpr.extend(runs_tpr)
        all_tnr_fine.extend(runs_tnr_fine)
        all_tnr_coarse.extend(runs_tnr_coarse)
        all_ctnr.extend(runs_ctnr)
        all_ftnr.extend(runs_ftnr)

    last_row.append(np.mean(all_acc))
    last_row.append(np.std(all_acc))
    last_row.append(np.mean(runs_tpr))
    last_row.append(np.std(runs_tpr))
    last_row.append(np.mean(runs_tnr_fine))
    last_row.append(np.std(runs_tnr_fine))
    last_row.append(np.mean(runs_tnr_coarse))
    last_row.append(np.std(runs_tnr_coarse))
    last_row.append(np.mean(all_ctnr))
    last_row.append(np.std(all_ctnr))
    last_row.append(np.mean(all_ftnr))
    last_row.append(np.std(all_ftnr))

    assert len(name_row) == len(last_row)

    with open(csv_file_name, 'w+') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow(name_row)
        csv_writer.writerow(last_row)
