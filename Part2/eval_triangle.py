# Based on code by Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry Vetrov, Andrew Gordon Wilson
# https://github.com/timgaripov/dnn-mode-connectivity

import argparse
import numpy as np
import os
import tabulate
import torch
import torch.nn.functional as F

import data
import models
import curves
import utils

parser = argparse.ArgumentParser(description='DNN triangle evaluation')
parser.add_argument('--dir', type=str, default='/tmp/eval', metavar='DIR',
                    help='directory (default: /tmp/eval)')

parser.add_argument('--num_points', type=int, default=61, metavar='N',
                    help='number of points in the triangle is be about (num_points^2)/2 (default: 61)')

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
parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default=None, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--curve', type=str, default=None, metavar='CURVE',
                    help='curve type to use (default: None)')

parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
                    help='checkpoint to eval (default: None)')

parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

os.makedirs(args.dir, exist_ok=True)

torch.backends.cudnn.benchmark = True

loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    args.transform,
    args.use_test,
    shuffle_train=False
)

architecture = getattr(models, args.model)
model = curves.TriangleNet(
    num_classes,
    architecture.curve,
    True,
    architecture_kwargs=architecture.kwargs,
)
model.to(device)
checkpoint = torch.load(args.ckpt, map_location=torch.device(device))
model.load_state_dict(checkpoint['model_state'])

criterion = F.cross_entropy
regularizer = curves.l2_regularizer(args.wd)

t_grid = np.linspace(0, 1, args.num_points)

t_points = []
for t1 in t_grid:
    for t2 in t_grid:
        if t1 + t2 <= 1:
            t_points.append([t1, t2, 1 - t1 - t2])
t_points = np.array(t_points)

T = t_points.shape[0]

tr_loss = np.zeros(T)
tr_nll = np.zeros(T)
tr_acc = np.zeros(T)
te_loss = np.zeros(T)
te_nll = np.zeros(T)
te_acc = np.zeros(T)
tr_err = np.zeros(T)
te_err = np.zeros(T)

columns = ['t', 'Train loss', 'Train nll', 'Train error (%)', 'Test nll', 'Test error (%)']

t = torch.FloatTensor([0.0, 0.0, 0.0]).to(device)

for i, t_value in enumerate(t_points):
    t.data = torch.FloatTensor(t_value)

    utils.update_bn(loaders['train'], model, device, t=t)
    tr_res = utils.test(loaders['train'], model, criterion, device, regularizer, t=t)
    te_res = utils.test(loaders['test'], model, criterion, device, regularizer, t=t)
    tr_loss[i] = tr_res['loss']
    tr_nll[i] = tr_res['nll']
    tr_acc[i] = tr_res['accuracy']
    tr_err[i] = 100.0 - tr_acc[i]
    te_loss[i] = te_res['loss']
    te_nll[i] = te_res['nll']
    te_acc[i] = te_res['accuracy']
    te_err[i] = 100.0 - te_acc[i]
    
    values = [t, tr_loss[i], tr_nll[i], tr_err[i], te_nll[i], te_err[i]]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
    if i % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)
    
def stats(values):
    min = np.min(values)
    max = np.max(values)
    avg = np.mean(values)
    return min, max, avg

tr_loss_min, tr_loss_max, tr_loss_avg = stats(tr_loss)
tr_nll_min, tr_nll_max, tr_nll_avg = stats(tr_nll)
tr_err_min, tr_err_max, tr_err_avg = stats(tr_err)

te_loss_min, te_loss_max, te_loss_avg = stats(te_loss)
te_nll_min, te_nll_max, te_nll_avg = stats(te_nll)
te_err_min, te_err_max, te_err_avg = stats(te_err)

print(tabulate.tabulate([
        ['train loss', tr_loss[0], tr_loss[-1], tr_loss_min, tr_loss_max, tr_loss_avg],
        ['train error (%)', tr_err[0], tr_err[-1], tr_err_min, tr_err_max, tr_err_avg],
        ['test nll', te_nll[0], te_nll[-1], te_nll_min, te_nll_max, te_nll_avg],
        ['test error (%)', te_err[0], te_err[-1], te_err_min, te_err_max, te_err_avg],
    ], [
        '', 'start', 'end', 'min', 'max', 'avg',
    ], tablefmt='simple', floatfmt='10.4f'))

np.savez(
    os.path.join(args.dir, args.ckpt.split('/')[-2]),
    ts=t_points,
    tr_loss=tr_loss,
    tr_loss_min=tr_loss_min,
    tr_loss_max=tr_loss_max,
    tr_loss_avg=tr_loss_avg,
    tr_nll=tr_nll,
    tr_nll_min=tr_nll_min,
    tr_nll_max=tr_nll_max,
    tr_nll_avg=tr_nll_avg,
    tr_acc=tr_acc,
    tr_err=tr_err,
    tr_err_min=tr_err_min,
    tr_err_max=tr_err_max,
    tr_err_avg=tr_err_avg,
    te_loss=te_loss,
    te_loss_min=te_loss_min,
    te_loss_max=te_loss_max,
    te_loss_avg=te_loss_avg,
    te_nll=te_nll,
    te_nll_min=te_nll_min,
    te_nll_max=te_nll_max,
    te_nll_avg=te_nll_avg,
    te_acc=te_acc,
    te_err=te_err,
    te_err_min=te_err_min,
    te_err_max=te_err_max,
    te_err_avg=te_err_avg,
)
