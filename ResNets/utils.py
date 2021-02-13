import numpy as np
import os
import random
import torch
import torchvision
# from scipy.stats import bernoulli

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from pathlib import Path


def int_quant(x):
    '''
    :param x: a torch tensor
    :return: a quantized torch tensor
    '''
    s = torch.sign(x)
    abs_vals = torch.abs(x)
    lower = torch.floor(abs_vals)
    upper = torch.ceil(abs_vals)
    prob = (abs_vals - lower) / (upper - lower + torch.finfo(float).eps)
    v = torch.bernoulli(prob)
    out = s * (lower + v)
    return out


def nat_compression(x):
    s = torch.sign(x)
    abs_vals = torch.abs(x)
    power = torch.log2(abs_vals)
    lower = torch.pow(2., torch.floor(power))
    upper = torch.pow(2., torch.floor(power) + 1)
    prob = (abs_vals - lower) / (upper - lower + torch.finfo(float).eps)
    v = torch.round(prob)
    return s * lower * torch.pow(2., v)


def load_data(dataset='cifar10', batch_size=128, num_workers=4):
    """
    Loads the required dataset
    :param dataset: Can be either 'cifar10' or 'cifar100'
    :param batch_size: The desired batch size
    :return: Tuple (train_loader, test_loader, num_classes)
    """
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if dataset == 'cifar10':
        # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        num_classes = 10
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100':
        num_classes = 100
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    else:
        raise ValueError('Only cifar 10 and cifar 100 are supported')

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader, num_classes


def test_accuracy(net, testloader, device):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def accuracy_and_loss(net, dataloader, device, criterion):
    net.eval()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels).cpu().item() / len(dataloader)

    return correct / total, loss


def save_results(losses, test_losses, train_acc, test_acc, it_train, it_test, grad_norms, max_coords=None,
                 max_sum_coords=None, method='sgd', lrs=None, experiment='lenet', n_mbs=None, alphas=None,
                 folder='cifar10_logs'):
    if lrs is None:
        lrs = []
    if max_coords is None:
        max_coords = []
    if max_sum_coords is None:
        max_sum_coords = []
    if n_mbs is None:
        n_mbs = []
    if alphas is None:
        alphas = []
    path_folder = './{0}/{1}/'.format(folder, experiment)
    Path(path_folder).mkdir(parents=True, exist_ok=True)
    path = path_folder + method + '_'
    to_save = [alphas, n_mbs, max_coords, max_sum_coords, losses, test_losses, train_acc, test_acc, it_train, it_test,
               grad_norms, lrs]
    prefixes = ['al', 'mb', 'mc', 'msc', 'l', 'tl', 'a', 'ta', 'itr', 'ite', 'gn', 'lr']
    for log, prefix in zip(to_save, prefixes):
        np.save(path + prefix + '.npy', log)


def load_results(method='sgd', experiment='lenet', folder='cifar10_logs', load_lr=False):
    path = './{0}/{1}/{2}_'.format(folder, experiment, method)
    prefixes = ['al', 'mb', 'mc', 'msc', 'l', 'tl', 'a', 'ta', 'itr', 'ite', 'gn']
    if load_lr:
        prefixes += ['lr']
    out = [np.load(path + prefix + '.npy') for prefix in prefixes]
    return tuple(out)


def dist_nets(net1, net2):
    dist = 0
    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        dist += (p1 - p2).norm().item() ** 2
    return np.sqrt(dist)


def net_weights_norm(net):
    weights_norm = 0
    for p in net.parameters():
        weights_norm += p.norm().item() ** 2
    return np.sqrt(weights_norm)


def smooth_array(x, k=2, skip=None):
    if skip is None:
        skip = k
    xs = [np.roll(x, i) for i in range(k)]
    for i in range(k):
        xs[i][0] = x[0]
    output = np.mean(xs, axis=0)
    output[:skip] = x[:skip]
    return output


def full_grad(net, optimizer, trainloader, device):
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    full_loss = 0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        full_loss += loss.cpu().item()
        total += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            p.grad.data /= total
    return correct / total, full_loss / total


def estim_full_grad(device, net, optimizer, trainloader, loss):
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    normalization_factor = len(trainloader)
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.replace_grad_estimate(normalization_factor)


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


# class Results_many_runs():
#     def __init__(self, label='', marker=','):
#         self.ave_loss = 0
#         self.ave_test_loss = 0
#         self.ave_train_acc = 0
#         self.ave_test_acc = 0
#         self.ave_grad_norm = 0
#         self.ave_max_coords = 0
#         self.ave_max_sum_coords = 0
#
#         self.std_loss = 0
#         self.std_test_loss = 0
#         self.std_train_acc = 0
#         self.std_test_acc = 0
#         self.std_grad_norm = 0
#         self.std_max_coords = 0
#         self.std_max_sum_coords = 0
#         self.label = label
#         self.marker = marker
#
#     def read_logs(self, method, experiment, n_runs=1, eps=0):
#         self.n_runs = n_runs
#         for run in range(0, n_runs):
#             method_ = method + '_run_{}'.format(run)
#             max_coords, max_sum_coords, loss, test_loss, train_acc, test_acc, self.it_train, self.it_test, grad_norm = load_results(
#                 method_, experiment=experiment)
#             if train_acc.size > 0 and np.max(train_acc) <= 1.:
#                 train_acc *= 100
#             if test_acc.size > 0 and np.max(test_acc) <= 1.:
#                 test_acc *= 100
#             if max_coords is not []:
#                 self.ave_max_coords += np.log(max_coords + eps) / n_runs
#                 self.std_max_coords += np.log(max_coords + eps) ** 2 / n_runs
#             if max_sum_coords is not []:
#                 self.ave_max_sum_coords += np.log(max_sum_coords + eps) / n_runs
#                 self.std_max_sum_coords += np.log(max_sum_coords + eps) ** 2 / n_runs
#             self.ave_loss += np.log(loss + eps) / n_runs
#             self.std_loss += np.log(loss + eps) ** 2 / n_runs
#             self.ave_test_loss += test_loss / n_runs
#             self.std_test_loss += test_loss ** 2 / n_runs
#             self.ave_train_acc += train_acc / n_runs
#             self.std_train_acc += train_acc ** 2 / n_runs
#             self.ave_test_acc += test_acc / n_runs
#             self.std_test_acc += test_acc ** 2 / n_runs
#             self.ave_grad_norm += np.log(grad_norm + eps) / n_runs
#             self.std_grad_norm += np.log(grad_norm + eps) ** 2 / n_runs
#
#         if len(self.ave_max_coords) > 1:
#             self.std_max_coords = np.sqrt(self.std_max_coords - self.ave_max_coords ** 2)
#             self.ave_max_coords = np.exp(self.ave_max_coords)
#         if len(self.ave_max_sum_coords) > 1:
#             self.std_max_sum_coords = np.sqrt(self.std_max_sum_coords - self.ave_max_sum_coords ** 2)
#             self.ave_max_sum_coords = np.exp(self.ave_max_sum_coords)
#         self.std_loss = np.sqrt(self.std_loss - self.ave_loss ** 2)
#         self.ave_loss = np.exp(self.ave_loss)
#         self.std_test_loss = np.sqrt(self.std_test_loss - self.ave_test_loss ** 2)
#         self.std_train_acc = np.sqrt(self.std_train_acc - self.ave_train_acc ** 2)
#         self.std_test_acc = np.sqrt(self.std_test_acc - self.ave_test_acc ** 2)
#         self.std_grad_norm = np.sqrt(self.std_grad_norm - self.ave_grad_norm ** 2)
#         self.ave_grad_norm = np.exp(self.ave_grad_norm)
#
#     def plot_train_acc(self, step=1, markevery=1, std=True, alpha=0.5, **kwargs):
#         plt.plot(self.it_train[::step], self.ave_train_acc[::step], label=self.label, marker=self.marker,
#                  markevery=markevery, **kwargs)
#         if std:
#             upper = self.ave_train_acc[::step] + self.std_train_acc[::step]
#             lower = self.ave_train_acc[::step] - self.std_train_acc[::step]
#             plt.fill_between(self.it_train[::step], upper, lower, alpha=alpha, **kwargs)
#             plt.ylabel('Train accuracy')
#
#     def plot_max_coords(self, step=1, markevery=1, std=True, alpha=0.5, **kwargs):
#         plt.plot(np.linspace(0, self.it_train[-1], len(self.ave_max_coords[::step])), self.ave_max_coords[::step],
#                  label=self.label,
#                  marker=self.marker,
#                  markevery=markevery, **kwargs)
#         if std:
#             upper = self.ave_max_coords[::step] + self.std_max_coords[::step]
#             lower = self.ave_max_coords[::step] - self.std_max_coords[::step]
#             plt.fill_between(np.linspace(0, self.it_train[-1], len(self.ave_max_coords[::step])), upper, lower,
#                              alpha=alpha, **kwargs)
#
#     def plot_max_sum_coords(self, step=1, markevery=1, std=True, alpha=0.5, **kwargs):
#         plt.plot(np.linspace(0, self.it_train[-1], len(self.ave_max_sum_coords[::step])),
#                  self.ave_max_sum_coords[::step],
#                  label=self.label,
#                  marker=self.marker,
#                  markevery=markevery, **kwargs)
#         if std:
#             upper = self.ave_max_sum_coords[::step] + self.std_max_sum_coords[::step]
#             lower = self.ave_max_sum_coords[::step] - self.std_max_sum_coords[::step]
#             plt.fill_between(np.linspace(0, self.it_train[-1], len(self.ave_max_sum_coords[::step])), upper, lower,
#                              alpha=alpha, **kwargs)
#
#     def plot_test_acc(self, step=1, markevery=1, std=True, alpha=0.5, **kwargs):
#         plt.plot(self.it_test[::step], self.ave_test_acc[::step], label=self.label, marker=self.marker,
#                  markevery=markevery, **kwargs)
#         if std:
#             upper = self.ave_test_acc[::step] + self.std_test_acc[::step]
#             lower = self.ave_test_acc[::step] - self.std_test_acc[::step]
#             plt.fill_between(self.it_test[::step], upper, lower, alpha=alpha, **kwargs)
#             plt.ylabel('Test accuracy')
#
#     def plot_test_loss(self, step=1, markevery=1, std=True, alpha=0.5, **kwargs):
#         plt.plot(self.it_test, self.ave_test_loss, label=self.label, marker=self.marker, markevery=markevery, **kwargs)
#         if std:
#             upper = np.exp(np.log(self.ave_test_loss[::step]) + self.std_test_loss[::step])
#             lower = np.exp(np.log(self.ave_test_loss[::step]) - self.std_test_loss[::step])
#             plt.fill_between(self.it_train[::step], upper, lower, alpha=alpha, **kwargs)
#             plt.ylabel('Test loss')
#
#     def plot_train_loss(self, step=1, markevery=1, std=True, alpha=0.5, **kwargs):
#         plt.plot(self.it_train[::step], self.ave_loss[::step], label=self.label, marker=self.marker,
#                  markevery=markevery, **kwargs)
#         if std:
#             upper = np.exp(np.log(self.ave_loss[::step]) + self.std_loss[::step])
#             lower = np.exp(np.log(self.ave_loss[::step]) - self.std_loss[::step])
#             plt.fill_between(self.it_train[::step], upper, lower, alpha=alpha, **kwargs)
#             plt.ylabel('Train loss')
#
#     def plot_grad_norm(self, step, markevery=1, std=True, alpha=0.5, **kwargs):
#         n_epoch = self.it_train[-1]
#         epochs = np.linspace(0, n_epoch, len(self.ave_grad_norm[::step]))
#         plt.plot(epochs, self.ave_grad_norm[::step], label=self.label, marker=self.marker, markevery=markevery,
#                  **kwargs)
#         if std:
#             upper = np.exp(np.log(self.ave_grad_norm[::step]) + self.std_grad_norm[::step])
#             lower = np.exp(np.log(self.ave_grad_norm[::step]) - self.std_grad_norm[::step])
#             plt.fill_between(epochs, upper, lower, alpha=alpha, **kwargs)
#             plt.ylabel('Gradient norm')

class Results_many_runs():
    def __init__(self, label='', marker=','):
        self.ave_loss = 0
        self.ave_test_loss = 0
        self.ave_train_acc = 0
        self.ave_test_acc = 0
        self.ave_grad_norm = 0
        self.ave_max_coords = 0
        self.ave_max_sum_coords = 0
        self.ave_n_mbs = 0
        self.ave_alphas = 0

        self.std_loss = 0
        self.std_test_loss = 0
        self.std_train_acc = 0
        self.std_test_acc = 0
        self.std_grad_norm = 0
        self.std_max_coords = 0
        self.std_max_sum_coords = 0
        self.std_alphas = 0
        self.label = label
        self.marker = marker

    def read_logs(self, method, experiment, n_runs=1, eps=0):
        self.n_runs = n_runs
        for run in range(0, n_runs):
            method_ = method + '_run_{}'.format(run)
            alphas, n_mbs, max_coords, max_sum_coords, loss, test_loss, train_acc, test_acc, self.it_train, self.it_test, grad_norm = load_results(
                method_, experiment=experiment)
            if train_acc.size > 0 and np.max(train_acc) <= 1.:
                train_acc *= 100
            if test_acc.size > 0 and np.max(test_acc) <= 1.:
                test_acc *= 100
            if max_coords is not []:
                self.ave_max_coords += np.log(max_coords + eps) / n_runs
                self.std_max_coords += np.log(max_coords + eps) ** 2 / n_runs
            if max_sum_coords is not []:
                self.ave_max_sum_coords += np.log(max_sum_coords + eps) / n_runs
                self.std_max_sum_coords += np.log(max_sum_coords + eps) ** 2 / n_runs

            min_len = min(len(loss), len(n_mbs))
            n_mbs = n_mbs[:min_len]
            loss = loss[:min_len]
            self.it_train = self.it_train[:min_len]

            self.ave_n_mbs += n_mbs / n_runs
            self.ave_loss += np.log(loss + eps) / n_runs
            self.std_loss += np.log(loss + eps) ** 2 / n_runs
            self.ave_test_loss += test_loss / n_runs
            self.std_test_loss += test_loss ** 2 / n_runs
            self.ave_alphas += alphas / n_runs
            self.std_alphas += alphas ** 2 / n_runs
            self.ave_train_acc += train_acc / n_runs
            self.std_train_acc += train_acc ** 2 / n_runs
            self.ave_test_acc += test_acc / n_runs
            self.std_test_acc += test_acc ** 2 / n_runs
            self.ave_grad_norm += np.log(grad_norm + eps) / n_runs
            self.std_grad_norm += np.log(grad_norm + eps) ** 2 / n_runs

        if len(self.ave_max_coords) > 1:
            self.std_max_coords = np.sqrt(self.std_max_coords - self.ave_max_coords ** 2)
            self.ave_max_coords = np.exp(self.ave_max_coords)
        if len(self.ave_max_sum_coords) > 1:
            self.std_max_sum_coords = np.sqrt(self.std_max_sum_coords - self.ave_max_sum_coords ** 2)
            self.ave_max_sum_coords = np.exp(self.ave_max_sum_coords)

        self.std_alphas = np.sqrt(self.std_alphas - self.ave_alphas ** 2)
        self.std_loss = np.sqrt(self.std_loss - self.ave_loss ** 2)
        self.ave_loss = np.exp(self.ave_loss)
        self.std_test_loss = np.sqrt(self.std_test_loss - self.ave_test_loss ** 2)
        self.std_train_acc = np.sqrt(self.std_train_acc - self.ave_train_acc ** 2)
        self.std_test_acc = np.sqrt(self.std_test_acc - self.ave_test_acc ** 2)
        self.std_grad_norm = np.sqrt(self.std_grad_norm - self.ave_grad_norm ** 2)
        self.ave_grad_norm = np.exp(self.ave_grad_norm)

    def plot_train_acc(self, step=1, markevery=1, std=True, alpha=0.5, **kwargs):
        plt.plot(self.it_train[::step], self.ave_train_acc[::step], label=self.label, marker=self.marker,
                 markevery=markevery, **kwargs)
        if std:
            upper = self.ave_train_acc[::step] + self.std_train_acc[::step]
            lower = self.ave_train_acc[::step] - self.std_train_acc[::step]
            plt.fill_between(self.it_train[::step], upper, lower, alpha=alpha, **kwargs)
            plt.ylabel('Train accuracy')

    def plot_alphas(self, step=10, markevery=1, std=True, alpha=0.5, **kwargs):
        plt.plot(np.linspace(0, self.it_train[-1], len(self.ave_alphas[::step])), self.ave_alphas[::step], label=self.label, marker=self.marker,
                 markevery=markevery, **kwargs)
        if std:
            upper = self.ave_alphas[::step] + self.std_alphas[::step]
            lower = self.ave_alphas[::step] - self.std_alphas[::step]
            plt.fill_between(np.linspace(0, self.it_train[-1], len(self.ave_alphas[::step])), upper, lower, alpha=alpha, **kwargs)
            plt.ylabel('Alphas')

    def plot_max_coords(self, step=1, markevery=1, std=True, alpha=0.5, **kwargs):
        plt.plot(np.linspace(0, self.it_train[-1], len(self.ave_max_coords[::step])), self.ave_max_coords[::step],
                 label=self.label,
                 marker=self.marker,
                 markevery=markevery, **kwargs)
        if std:
            upper = self.ave_max_coords[::step] + self.std_max_coords[::step]
            lower = self.ave_max_coords[::step] - self.std_max_coords[::step]
            plt.fill_between(np.linspace(0, self.it_train[-1], len(self.ave_max_coords[::step])), upper, lower,
                             alpha=alpha, **kwargs)

    def plot_max_sum_coords(self, step=1, markevery=1, std=True, alpha=0.5, **kwargs):
        plt.plot(np.linspace(0, self.it_train[-1], len(self.ave_max_sum_coords[::step])),
                 self.ave_max_sum_coords[::step],
                 label=self.label,
                 marker=self.marker,
                 markevery=markevery, **kwargs)
        if std:
            upper = self.ave_max_sum_coords[::step] + self.std_max_sum_coords[::step]
            lower = self.ave_max_sum_coords[::step] - self.std_max_sum_coords[::step]
            plt.fill_between(np.linspace(0, self.it_train[-1], len(self.ave_max_sum_coords[::step])), upper, lower,
                             alpha=alpha, **kwargs)

    def plot_test_acc(self, step=1, markevery=1, std=True, alpha=0.5, **kwargs):
        step *= len(self.it_test) // len(self.it_train)
        ave_test_acc = self.ave_test_acc[::step]
        std_test_acc = self.std_test_acc[::step]
        min_len = min(len(self.ave_n_mbs), len(ave_test_acc))
        plt.plot(self.ave_n_mbs[:min_len], ave_test_acc[:min_len], label=self.label, marker=self.marker,
                 markevery=markevery, **kwargs)
        if std:
            upper = ave_test_acc[:min_len] + std_test_acc[:min_len]
            lower = ave_test_acc[:min_len] - std_test_acc[:min_len]
            plt.fill_between(self.ave_n_mbs[:min_len], upper, lower, alpha=alpha, **kwargs)
            plt.ylabel('Test accuracy')

    def plot_test_loss(self, step=1, markevery=1, std=True, alpha=0.5, **kwargs):
        plt.plot(self.it_test, self.ave_test_loss, label=self.label, marker=self.marker, markevery=markevery, **kwargs)
        if std:
            upper = np.exp(np.log(self.ave_test_loss[::step]) + self.std_test_loss[::step])
            lower = np.exp(np.log(self.ave_test_loss[::step]) - self.std_test_loss[::step])
            plt.fill_between(self.it_train[::step], upper, lower, alpha=alpha, **kwargs)
            plt.ylabel('Test loss')

    def plot_train_loss(self, step=1, markevery=1, std=True, alpha=0.5, **kwargs):
        plt.plot(self.ave_n_mbs[::step], self.ave_loss[::step], label=self.label, marker=self.marker,
                 markevery=markevery, **kwargs)
        if std:
            upper = np.exp(np.log(self.ave_loss[::step]) + self.std_loss[::step])
            lower = np.exp(np.log(self.ave_loss[::step]) - self.std_loss[::step])
            plt.fill_between(self.ave_n_mbs[::step], upper, lower, alpha=alpha, **kwargs)
        plt.ylabel('Train loss')

        # plt.annotate(text=text_str, xy=(self.ave_n_mbs[-1], self.ave_loss[-1]),
        #              xytext=(self.ave_n_mbs[-1]/2, self.ave_loss[-1] + 0.1),
        #                 arrowprops=dict(arrowstyle='<-', color='black'), fontsize=10)


#     def plot_grad_norm(self, step, markevery=1, std=True, alpha=0.5, **kwargs):
#         n_epoch = self.it_train[-1]
#         epochs = np.linspace(0, n_epoch, len(self.ave_grad_norm[::step]))
#         plt.plot(epochs, self.ave_grad_norm[::step], label=self.label, marker=self.marker, markevery=markevery,
#                  **kwargs)
#         if std:
#             upper = np.exp(np.log(self.ave_grad_norm[::step]) + self.std_grad_norm[::step])
#             lower = np.exp(np.log(self.ave_grad_norm[::step]) - self.std_grad_norm[::step])
#             plt.fill_between(epochs, upper, lower, alpha=alpha, **kwargs)
#             plt.ylabel('Gradient norm')

def seed_everything(seed=1029):
    '''
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
