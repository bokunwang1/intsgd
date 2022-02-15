import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib.ticker import MaxNLocator
import os

sns.set(style="whitegrid", context="talk", font_scale=1.2, palette=sns.color_palette("bright"), color_codes=False)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.figsize'] = (8, 6)

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['mathtext.fontset'] = 'cm'


def load_results(seed, n_workers, method='ExactReducer', experiment='ResNet18', folder='output'):
    path = './{0}/{1}-{2}/{3}_{4}_'.format(folder, experiment, n_workers, seed, method)
    prefixes = ['mbs', 'te_l', 'te_acc', 'al']
    out = [np.load(path + prefix + '.npy') for prefix in prefixes]
    return tuple(out)


class Results_many_runs():
    def __init__(self, label='', marker=','):
        self.ave_test_loss = 0
        self.ave_test_acc = 0
        self.ave_n_mbs = 0
        self.ave_alphas = 0
        self.max_alphas = 0
        self.max_alphas_p = 0
        self.max_alphas_q = 0
        self.max_per_overflow = 0
        self.min_alphas = 0
        self.label = label
        self.marker = marker

    def read_logs(self, method, experiment, seeds, n_workers, eps=0.0):
        n_runs = len(seeds)
        all_n_mbs = []
        all_test_loss = []
        all_alphas = []
        all_test_acc = []
        for run, seed in enumerate(seeds):
            n_mbs, test_loss, test_acc, alphas = load_results(seed=seed, n_workers=n_workers,
                                                              method=method,
                                                              experiment=experiment)
            if test_acc.size > 0 and np.max(test_acc) <= 1.:
                test_acc *= 100
            all_n_mbs.append(n_mbs)
            all_test_loss.append(test_loss)
            all_test_acc.append(test_acc)
            all_alphas.append(alphas)

        self.ave_n_mbs = np.mean(all_n_mbs, axis=0)
        self.ave_test_loss = np.mean(all_test_loss, axis=0)
        self.ave_test_acc = np.mean(all_test_acc, axis=0)
        self.ave_alphas = np.mean(all_alphas, axis=0)
        self.max_test_acc = np.max(all_test_acc, axis=0)
        self.min_test_acc = np.min(all_test_acc, axis=0)
        self.max_test_loss = np.max(all_test_loss, axis=0)
        self.min_test_loss = np.min(all_test_loss, axis=0)
        self.max_alphas = np.max(all_alphas, axis=0)
        self.min_alphas = np.min(all_alphas, axis=0)

        self.std_test_acc = np.std(all_test_acc, axis=0)
        self.std_test_loss = np.std(all_test_loss, axis=0)

    def plot_alphas(self, step=1, markevery=1, std=True, alpha=0.5, **kwargs):
        if std:
            lower = self.min_alphas
            upper = self.max_alphas
            plt.fill_between(np.arange(len(lower[::step])), upper, lower, alpha=alpha,
                             **kwargs)
        ave_alphas = self.ave_alphas
        plt.plot(np.arange(len(ave_alphas[::step])), ave_alphas[::step],
                 label=self.label, marker=self.marker, markevery=markevery, **kwargs)

    def plot_test_acc(self, markevery=1, std=True, alpha=0.5, **kwargs):
        if std:
            # upper = self.ave_test_acc + self.std_test_acc
            # lower = self.ave_test_acc - self.std_test_acc
            upper = self.max_test_acc
            lower = self.min_test_acc
            plt.fill_between(np.arange(len(upper)), upper, lower, alpha=alpha, **kwargs)
            plt.ylabel('Test accuracy')
        plt.plot(np.arange(len(self.ave_test_acc)), self.ave_test_acc, label=self.label, marker=self.marker,
                 markevery=markevery,
                 **kwargs)
        print("{0}: Final test acc: {1}, std: {2}".format(self.label, self.ave_test_acc[-1], self.std_test_acc[-1]))

    def plot_test_loss(self, markevery=1, std=True, alpha=0.5, **kwargs):
        if std:
            # upper = np.exp(self.ave_test_loss_log + self.std_test_loss_log)
            # lower = np.exp(self.ave_test_loss_log - self.std_test_loss_log)
            upper = self.max_test_loss
            lower = self.min_test_loss
            plt.fill_between(np.arange(len(upper)), upper, lower, alpha=alpha, **kwargs)
            plt.ylabel('Test loss')

        ave_test_loss = self.ave_test_loss
        plt.plot(np.arange(len(ave_test_loss)), ave_test_loss, label=self.label, marker=self.marker,
                 markevery=markevery, **kwargs)
        print("{0}: Final test error: {1}, std: {2}".format(self.label, ave_test_loss[-1],
                                                            self.std_test_loss[-1]))
