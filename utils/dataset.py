import torch
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler


# Dataloader for MNIST
def mnist(batch_size, data_augmentation = True, shuffle = True, valid_ratio = None):

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

    trainset = datasets.MNIST('./data/MNIST', train = True, download = True, transform = transform)
    validset = datasets.MNIST('./data/MNIST', train = True, download = True, transform = transform)
    testset = datasets.MNIST('./data/MNIST', train = False, download = True, transform = transform)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    if valid_ratio is not None and valid_ratio > 0.:

        instance_num = len(trainset)
        indices = list(range(instance_num))
        split_pt = int(instance_num * valid_ratio)

        train_idx, valid_idx = indices[split_pt:], indices[:split_pt]
        train_sampler, valid_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(valid_idx)
        assert shuffle == True, 'shuffle must be true with a validation set'

        train_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, sampler = train_sampler, num_workers = 1, pin_memory = True)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size = batch_size, sampler = valid_sampler, num_workers = 1, pin_memory = True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 1, pin_memory = True)

        return train_loader, valid_loader, test_loader, classes

    else:

        train_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = shuffle, num_workers = 1, pin_memory = True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 1, pin_memory = True)

        return train_loader, None, test_loader, classes

# Dataloader for FashionMNIST
def FashionMNIST(batch_size, data_augmentation = True, shuffle = True, valid_ratio = None):

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

    trainset = datasets.FashionMNIST('./data/FashionMNIST', train = True, download = True, transform = transform)
    validset = datasets.FashionMNIST('./data/FashionMNIST', train = True, download = True, transform = transform)
    testset = datasets.FashionMNIST('./data/FashionMNIST', train = False, download = True, transform = transform)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    if valid_ratio is not None and valid_ratio > 0.:

        instance_num = len(trainset)
        indices = list(range(instance_num))
        split_pt = int(instance_num * valid_ratio)

        train_idx, valid_idx = indices[split_pt:], indices[:split_pt]
        train_sampler, valid_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(valid_idx)
        assert shuffle == True, 'shuffle must be true with a validation set'

        train_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, sampler = train_sampler, num_workers = 1, pin_memory = True)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size = batch_size, sampler = valid_sampler, num_workers = 1, pin_memory = True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 1, pin_memory = True)

        return train_loader, valid_loader, test_loader, classes

    else:

        train_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = shuffle, num_workers = 1, pin_memory = True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 1, pin_memory = True)

        return train_loader, None, test_loader, classes

# Dataloader for CIFAR10
def CIFAR10(batch_size, data_augmentation = True, shuffle = True, valid_ratio = None):

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10('./data/CIFAR10', train = True, download = True, transform = transform)
    validset = datasets.CIFAR10('./data/CIFAR10', train = True, download = True, transform = transform)
    testset = datasets.CIFAR10('./data/CIFAR10', train = False, download = True, transform = transform)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    if valid_ratio is not None and valid_ratio > 0.:

        instance_num = len(trainset)
        indices = list(range(instance_num))
        split_pt = int(instance_num * valid_ratio)

        train_idx, valid_idx = indices[split_pt:], indices[:split_pt]
        train_sampler, valid_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(valid_idx)
        assert shuffle == True, 'shuffle must be true with a validation set'

        train_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, sampler = train_sampler, num_workers = 1, pin_memory = True)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size = batch_size, sampler = valid_sampler, num_workers = 1, pin_memory = True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 1, pin_memory = True)

        return train_loader, valid_loader, test_loader, classes

    else:

        train_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = shuffle, num_workers = 1, pin_memory = True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 1, pin_memory = True)

        return train_loader, None, test_loader, classes
