# Based on code by Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry Vetrov, Andrew Gordon Wilson
# https://github.com/timgaripov/dnn-mode-connectivity

import os
import torch
import torchvision
import torchvision.transforms as transforms


class Transforms:

    class FashionMNIST:
         
        class LeNet:
            
            train = transforms.Compose([
                transforms.ToTensor(),
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
            ])

    class CIFAR10:

        class VGG:

            train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        class ResNet:

            train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])

        class ResNet_like:

            train = transforms.Compose([
                transforms.ToTensor(),
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
            ])

    CIFAR100 = CIFAR10
    
    class SVHN:
        
        class ResNet_like:

            train = transforms.Compose([
                transforms.ToTensor(),
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
            ])        


def loaders(dataset, path, batch_size, num_workers, transform_name, use_test=False,
            shuffle_train=True):
    ds = getattr(torchvision.datasets, dataset)
    path = os.path.join(path, dataset.lower())
    transform = getattr(getattr(Transforms, dataset), transform_name)
    if dataset == 'SVHN':
        train_set = ds(path, split='train', download=True, transform=transform.train)
    else:
        train_set = ds(path, train=True, download=True, transform=transform.train)

    if use_test:
        print('You are going to run models on the test set. Are you sure?')
        if dataset == 'SVHN':
            test_set = ds(path, split='test', download=True, transform=transform.test)
        else:
            test_set = ds(path, train=False, download=True, transform=transform.test)
    else:
        print("Using train + validation (5000)")
        train_set.data = train_set.data[:-5000]
        if dataset == 'SVHN':
            train_set.labels = train_set.labels[:-5000]
            test_set = ds(path, split='test', download=True, transform=transform.test)
        else:
            train_set.targets = train_set.targets[:-5000]
            test_set = ds(path, train=True, download=True, transform=transform.test)

        test_set.train = False
        test_set.data = test_set.data[-5000:]
        if dataset == 'SVHN':
            test_set.labels = test_set.labels[-5000:]
        else:
            test_set.targets = test_set.targets[-5000:]

    if dataset == 'SVHN':
        num_classes = max(train_set.labels) + 1
    else:
        num_classes = max(train_set.targets) + 1
    num_classes = int(num_classes)

    return {
               'train': torch.utils.data.DataLoader(
                   train_set,
                   batch_size=batch_size,
                   shuffle=shuffle_train,
                   num_workers=num_workers,
                   pin_memory=True
               ),
               'test': torch.utils.data.DataLoader(
                   test_set,
                   batch_size=batch_size,
                   shuffle=False,
                   num_workers=num_workers,
                   pin_memory=True
               ),
           }, num_classes

