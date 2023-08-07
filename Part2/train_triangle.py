# Based on code by Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry Vetrov, Andrew Gordon Wilson
# https://github.com/timgaripov/dnn-mode-connectivity

import argparse
import os
import sys
import tabulate
import time
import torch
import torch.nn.functional as F

import curves
import data
import models
import utils
import pandas as pd

parser = argparse.ArgumentParser(description='DNN triangle training')
parser.add_argument('--dir', type=str, default='/tmp/curve/', metavar='DIR',
                    help='training directory (default: /tmp/curve/)')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true',
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default=None, metavar='MODEL', required=True,
                    help='model name (default: None)')

parser.add_argument('--init_curve_0', type=str, default=None, metavar='CKPT', required=True,
                    help='checkpoint to init vertex (edge) 1 (default: None)')
parser.add_argument('--init_curve_1', type=str, default=None, metavar='CKPT', required=True,
                    help='checkpoint to init vertex (edge) 2 (default: None)')
parser.add_argument('--init_curve_2', type=str, default=None, metavar='CKPT',required=True,
                    help='checkpoint to init vertex (edge) 3 (default: None)')

parser.add_argument('--fix_centers', dest='fix_centers', action='store_true',
                    help='fix edge centers (default: off)')

parser.set_defaults(init_edge_linear=True)
parser.add_argument('--init_edge_linear_off', dest='init_edge_linear', action='store_false',
                    help='turns off linear initialization of edge centers (default: on)')
parser.set_defaults(init_triangle_linear=True)
parser.add_argument('--init_triangle_linear_off', dest='init_triangle_linear', action='store_false',
                    help='turns off linear initialization of triangle center (default: on)')

parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=50, metavar='N',
                    help='save frequency (default: 50)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()

init_paths = [args.init_curve_0, args.init_curve_1, args.init_curve_2]

index = []
for ckpt_path in init_paths:
    s1, s2 = ckpt_path.split('/')[-2].split('_')[2:4]
    index.append([int(s1), int(s2)])
seeds = set()
for x in index:
    if args.fix_centers:
        seeds.add('-'.join(map(str, x)))
    else:
        seeds.add(str(x[0]))
        seeds.add(str(x[1]))
seeds = sorted(list(seeds))

args.dir = os.path.join(args.dir, '_'.join([args.model, args.dataset, \
                                            '_'.join(seeds), 'triangle', str(args.seed)]))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')
    
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    args.transform,
    args.use_test
)

architecture = getattr(models, args.model)

model = curves.TriangleNet(
    num_classes,
    architecture.curve,
    args.fix_centers,
    architecture_kwargs=architecture.kwargs,
)

base_model = None
for i, path in enumerate(init_paths):
    min_index = index[i]
    if base_model is None:
        base_model = curves.CurveNet(num_classes,
                                     getattr(curves, 'PolyChain'),
                                     architecture.curve,
                                     3,
                                     architecture_kwargs=architecture.kwargs,)
    checkpoint = torch.load(path)
    print('Loading %s as point #%d' % (path, i))
    base_model.load_state_dict(checkpoint['model_state'])
    model.import_base_parameters(base_model, min_index)
    
model.init_linear(init_edge_centers=args.init_edge_linear, init_triangle_center=args.init_triangle_linear)

model = model.to(device)

def learning_rate_schedule(base_lr, epoch, total_epochs):
    alpha = epoch / total_epochs
    if alpha <= 0.5:
        factor = 1.0
    elif alpha <= 0.9:
        factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
    else:
        factor = 0.01
    return factor * base_lr

criterion = F.cross_entropy
regularizer = curves.l2_regularizer(args.wd)
optimizer = torch.optim.SGD(
    filter(lambda param: param.requires_grad, model.parameters()),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=0.0
)

start_epoch = 1

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_nll', 'te_acc', 'time']

utils.save_checkpoint(
    args.dir,
    start_epoch - 1,
    model_state=model.state_dict(),
    optimizer_state=optimizer.state_dict()
)

train_stats = []

has_bn = utils.check_bn(model)
test_res = {'loss': None, 'accuracy': None, 'nll': None}
for epoch in range(start_epoch, args.epochs + 1):
    time_ep = time.time()

    lr = learning_rate_schedule(args.lr, epoch, args.epochs)
    utils.adjust_learning_rate(optimizer, lr)

    train_res = utils.train(loaders['train'], model, optimizer, criterion, device, regularizer)
    if not has_bn:
        test_res = utils.test(loaders['test'], model, criterion, device, regularizer)

    if epoch % args.save_freq == 0:
        utils.save_checkpoint(
            args.dir,
            epoch,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict()
        )

    time_ep = time.time() - time_ep
    values = [epoch, lr, train_res['loss'], train_res['accuracy'], test_res['nll'],
              test_res['accuracy'], time_ep]
    train_stats.append(values)

    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
    if epoch % 40 == 1 or epoch == start_epoch:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)
    
train_stats_df = pd.DataFrame(train_stats, columns=['epoch', 'lr', 'train_loss', 'train_acc', 'test_nll', 'test_acc','time_ep'])
train_stats_df.to_csv(os.path.join(args.dir, 'train_stats.csv'))

if args.epochs % args.save_freq != 0:
    utils.save_checkpoint(
        args.dir,
        args.epochs,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict()
    )

