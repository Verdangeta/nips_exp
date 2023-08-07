# Based on code by Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry Vetrov, Andrew Gordon Wilson
# https://github.com/timgaripov/dnn-mode-connectivity

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from functools import reduce
import operator

import curves

__all__ = ['ResNet9']


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self.cache = torch.zeros(1, 1, 1, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        if np.product(x.shape[1:]) != np.product(self.cache.shape[1:]):
            self.cache = x
            return self.cache

        self.cache = self.relu(x + self.cache)
        return self.cache
        

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, input):
        return input.reshape(input.shape[0], -1)

    
class ResNetBase(nn.Module):
    def __init__(self, num_classes, mode='resnet20', width=1, variance=None):
        super(ResNetBase, self).__init__()

        self.ReLU = nn.ReLU()
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.Flatten = Flatten()
        self.Identity = Identity()

        inp, oup = 3, num_classes
        
        if mode == 'resnet20':
            self.net = nn.Sequential(           

                 nn.Conv2d(inp, int(16 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU, self.Identity,

                 #1.1
                 nn.Conv2d(int(16 * width), int(16 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(16 * width), int(16 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.Identity,

                 #1.2
                 nn.Conv2d(int(16 * width), int(16 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(16 * width), int(16 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.Identity,


                 #1.3
                 nn.Conv2d(int(16 * width), int(16 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(16 * width), int(32 * width), kernel_size=3, stride=2, padding=1, bias=True),
                 self.Identity,


                 #2.1
                 nn.Conv2d(int(32 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(32 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.Identity,

                 #2.2
                 nn.Conv2d(int(32 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(32 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.Identity,

                 #2.3
                 nn.Conv2d(int(32 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(32 * width), int(64 * width), kernel_size=3, stride=2, padding=1, bias=True),
                 self.Identity,



                 #3.1
                 nn.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.Identity,

                 #3.2
                 nn.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.Identity,


                 #3.3
                 nn.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.Identity,

                 self.GAP,
                 self.Flatten,

                 nn.Linear(int(64 * width), oup))
        

        elif mode == 'resnet14':
            self.net = nn.Sequential(           

                 nn.Conv2d(inp, int(16 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU, self.Identity,

                 #1.1
                 nn.Conv2d(int(16 * width), int(16 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(16 * width), int(16 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.Identity,


                 #1.2
                 nn.Conv2d(int(16 * width), int(16 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(16 * width), int(32 * width), kernel_size=3, stride=2, padding=1, bias=True),
                 self.Identity,


                 #2.1
                 nn.Conv2d(int(32 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(32 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.Identity,


                 #2.3
                 nn.Conv2d(int(32 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(32 * width), int(64 * width), kernel_size=3, stride=2, padding=1, bias=True),
                 self.Identity,


                 #3.1
                 nn.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.Identity,

                 #3.3
                 nn.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.Identity,

                 self.GAP,
                 self.Flatten,

                 nn.Linear(int(64 * width), oup))

            
        elif mode == 'resnet9':
            self.net = nn.Sequential(           

                 nn.Conv2d(inp, int(16 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU, self.Identity,

                 #1.1
                 nn.Conv2d(int(16 * width), int(16 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(16 * width), int(16 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.Identity,
                
                 nn.MaxPool2d(2, 2),
                 nn.Conv2d(int(16 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU, self.Identity, 


                 #2.1
                 nn.Conv2d(int(32 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(32 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.Identity,
                
                 nn.MaxPool2d(2, 2),
                 nn.Conv2d(int(32 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU, self.Identity,


                 #3.1
                 nn.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.Identity,
                    
                 nn.MaxPool2d(2, 2),
                 
                #4.1
                 self.Identity,
                 nn.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.ReLU,

                 nn.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True),
                 self.Identity,
                 nn.MaxPool2d(2, 2),
                 
                 nn.BatchNorm2d(int(64 * width)),

                 self.GAP,
                 self.Flatten,

                 nn.Linear(int(64 * width), oup))

            
            
        for layer in self.net.children():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                if not isinstance(variance, type(None)):
                    nn.init.kaiming_normal_(layer.weight.data, 0.0, variance)
                else:
                    nn.init.kaiming_normal_(layer.weight)

                nn.init.constant_(layer.bias.data, 0.01)
    
    def forward(self, x):
        return self.net(x)


class ResNetCurve(nn.Module):
    def __init__(self, num_classes, fix_points, mode='resnet20', width=1, variance=None):
        super(ResNetCurve, self).__init__()

        self.ReLU = nn.ReLU()
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.Flatten = Flatten()
        self.Identity = Identity()

        inp, oup = 3, num_classes
        
        if mode == 'resnet20':
            self.net = nn.Sequential(           

                 curves.Conv2d(inp, int(16 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.ReLU, self.Identity,

                 #1.1
                 curves.Conv2d(int(16 * width), int(16 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.ReLU,

                 curves.Conv2d(int(16 * width), int(16 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.Identity,

                 #1.2
                 curves.Conv2d(int(16 * width), int(16 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.ReLU,

                 curves.Conv2d(int(16 * width), int(16 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.Identity,


                 #1.3
                 curves.Conv2d(int(16 * width), int(16 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.ReLU,

                 curves.Conv2d(int(16 * width), int(32 * width), kernel_size=3, stride=2, padding=1, bias=True, fix_points=fix_points),
                 self.Identity,


                 #2.1
                 curves.Conv2d(int(32 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.ReLU,

                 curves.Conv2d(int(32 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.Identity,

                 #2.2
                 curves.Conv2d(int(32 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.ReLU,

                 curves.Conv2d(int(32 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.Identity,

                 #2.3
                 curves.Conv2d(int(32 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.ReLU,

                 curves.Conv2d(int(32 * width), int(64 * width), kernel_size=3, stride=2, padding=1, bias=True, fix_points=fix_points),
                 self.Identity,



                 #3.1
                 curves.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.ReLU,

                 curves.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.Identity,

                 #3.2
                 curves.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.ReLU,

                 curves.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.Identity,


                 #3.3
                 curves.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.ReLU,

                 curves.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.Identity,

                 self.GAP,
                 self.Flatten,

                 curves.Linear(int(64 * width), oup, fix_points=fix_points))
        

        elif mode == 'resnet14':
            self.net = nn.Sequential(           

                 curves.Conv2d(inp, int(16 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.ReLU, self.Identity,

                 #1.1
                 curves.Conv2d(int(16 * width), int(16 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.ReLU,

                 curves.Conv2d(int(16 * width), int(16 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.Identity,


                 #1.2
                 curves.Conv2d(int(16 * width), int(16 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.ReLU,

                 curves.Conv2d(int(16 * width), int(32 * width), kernel_size=3, stride=2, padding=1, bias=True, fix_points=fix_points),
                 self.Identity,


                 #2.1
                 curves.Conv2d(int(32 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.ReLU,

                 curves.Conv2d(int(32 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.Identity,


                 #2.3
                 curves.Conv2d(int(32 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.ReLU,

                 curves.Conv2d(int(32 * width), int(64 * width), kernel_size=3, stride=2, padding=1, bias=True, fix_points=fix_points),
                 self.Identity,


                 #3.1
                 curves.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.ReLU,

                 curves.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.Identity,

                 #3.3
                 curves.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.ReLU,

                 curves.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.Identity,

                 self.GAP,
                 self.Flatten,

                 curves.Linear(int(64 * width), oup, fix_points=fix_points))

            
        elif mode == 'resnet9':
            self.net = nn.Sequential(           

                 curves.Conv2d(inp, int(16 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.ReLU, self.Identity,

                 #1.1
                 curves.Conv2d(int(16 * width), int(16 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.ReLU,

                 curves.Conv2d(int(16 * width), int(16 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.Identity,
                
                 nn.MaxPool2d(2, 2),
                 curves.Conv2d(int(16 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.ReLU, self.Identity, 


                 #2.1
                 curves.Conv2d(int(32 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.ReLU,

                 curves.Conv2d(int(32 * width), int(32 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.Identity,
                
                 nn.MaxPool2d(2, 2),
                 curves.Conv2d(int(32 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.ReLU, self.Identity,


                 #3.1
                 curves.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.ReLU,

                 curves.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.Identity,
                    
                 nn.MaxPool2d(2, 2),
                 
                #4.1
                 self.Identity,
                 curves.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.ReLU,

                 curves.Conv2d(int(64 * width), int(64 * width), kernel_size=3, stride=1, padding=1, bias=True, fix_points=fix_points),
                 self.Identity,
                 nn.MaxPool2d(2, 2),
                 
                 curves.BatchNorm2d(int(64 * width), fix_points=fix_points),

                
                 self.GAP,
                 self.Flatten,

                 curves.Linear(int(64 * width), oup, fix_points=fix_points))
            
            
        for layer in self.net.children():
            if isinstance(layer, curves.Linear) or isinstance(layer, curves.Conv2d):
                for i in range(layer.num_bends):
                    if not isinstance(variance, type(None)):
                        nn.init.kaiming_normal_(getattr(layer, 'weight_%d' % i).data, 0.0, variance)
                    else:
                        nn.init.kaiming_normal_(getattr(layer, 'weight_%d' % i))

                    nn.init.constant_(getattr(layer, 'bias_%d' % i).data, 0.01)


    def forward(self, x, coeffs_t):
        for layer in self.net:
            if issubclass(layer.__class__, curves.CurveModule):
                x = layer(x, coeffs_t)
            else:
                x = layer(x)
        return x


class ResNet9:
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {'mode': 'resnet9'}




