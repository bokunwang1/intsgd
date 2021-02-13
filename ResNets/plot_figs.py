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

import argparse
from utils import Results_many_runs
from config import seeds, lr_decay_epoches

parser = argparse.ArgumentParser(description='Plotting the figures')
parser.add_argument('--algs', nargs='+', dest='algo_list', help='List of algorithms')
parser.add_argument('--sigma_sqs', nargs='+', dest='sigma_sq_list', help='List of sigma_sqs')
parser.add_argument('--betas', nargs='+', dest='beta_list', help='List of values of beta')
parser.add_argument('--lw', nargs='+', dest='layerwise_list', help='List of layerwise or not')
parser.add_argument('--rr', nargs='+', dest='rr_list', help='List of rand round or not')
parser.add_argument('--alpha0s', nargs='+', dest='alpha0_list', help='List of alpha_coefs')
parser.add_argument('--lr0', action='store', dest='lr0', type=float, help='Value of initial stepsize')
parser.add_argument('--bs', action='store', dest='batch_size', type=int, help='Batch size for stochastic algorithms')
parser.add_argument('--net', action='store', dest='experiment', type=str, help='lenet, resnet or densenet')

args = parser.parse_args()
algo_list = args.algo_list
beta_list = args.beta_list
sigma_sq_list = args.sigma_sq_list
experiment = args.experiment
lr0 = args.lr0
batch_size = args.batch_size
layerwise_list = args.layerwise_list
rr_list = args.rr_list
alpha0_list = args.alpha0_list

epoch_lrs = [(lr_decay_epoches[0], lr0 / 10), (lr_decay_epoches[1], lr0 / 100)]
colors = ['r', 'y', 'm', 'b', 'g', 'c']
markers = ['D', 'o', 'x', '*', 'v', '.']

if len(algo_list) != len(beta_list) or len(beta_list) != len(sigma_sq_list):
    raise ValueError('The length of algorithm list should be equal to beta list! They are one-one corresponded.')

f1 = plt.figure()
eps = 0
step = 1
alpha = 0.3
plt.figure(figsize=(8, 6))

## Plot training loss
for i, alg in enumerate(algo_list):
    beta = beta_list[i]
    sigma_sq = sigma_sq_list[i]
    layerwise = layerwise_list[i]
    random_round = rr_list[i]
    alpha0 = alpha0_list[i]
    if alg == 'IntSGD':
        method = '{0}_{1}_int_layerwise_{2}_rand_round_{3}_beta_{4}_sigma_sq_{5}_lr_{6}'.format(experiment, alpha0,
                                                                                                layerwise, random_round,
                                                                                                beta, sigma_sq, lr0)
        for (epoch, lr) in epoch_lrs:
            method += '_{}_{}'.format(epoch, lr)

        # int_label = 'IntSGD' + '(' + r'$\alpha_0=$' + str(
        #     alpha0) + r'$, \beta=$' + str(float(beta)) + r'$,\sigma_Q^2=$' + str(sigma_sq)
        int_label = 'IntSGD' + '(' + r'$\beta=$' + str(float(beta))

        if layerwise == 'True':
            int_label += ',lw'
        else:
            int_label += ''

        if random_round == 'True':
            int_label += ',rnd'
        else:
            int_label += ''

        int_label += ')'

        res_int = Results_many_runs(label=int_label, marker=markers[i])
        res_int.read_logs(method, experiment, n_runs=len(seeds), eps=eps)
        res_int.plot_train_loss(step=step, alpha=alpha, markevery=10, color=colors[i])
    elif alg == 'SGD':
        method = '{0}_{1}_sgd_lr_{2}'.format(experiment, str(batch_size), lr0)
        for (epoch, lr) in epoch_lrs:
            method += '_{}_{}'.format(epoch, lr)
        res_sgd = Results_many_runs(label='SGD', marker=markers[i])
        res_sgd.read_logs(method, experiment, n_runs=len(seeds), eps=eps)
        res_sgd.plot_train_loss(step=step, alpha=alpha, markevery=10, color=colors[i])
        plt.axhline(y=res_sgd.ave_loss[-1], color='black', linestyle='--')
    elif alg == 'NatSGD':
        method = '{0}_{1}_nat_sgd_lr_{2}'.format(experiment, str(batch_size), lr0)
        for (epoch, lr) in epoch_lrs:
            method += '_{}_{}'.format(epoch, lr)
        res_nat = Results_many_runs(label='NatSGD', marker=markers[i])
        res_nat.read_logs(method, experiment, n_runs=len(seeds), eps=eps)
        res_nat.plot_train_loss(step=step, alpha=alpha, markevery=10, color=colors[i])

    elif alg == 'HintSGD':
        method = '{0}_{1}_hint_sgd_lr_{2}'.format(experiment, str(batch_size), lr0)
        for (epoch, lr) in epoch_lrs:
            method += '_{}_{}'.format(epoch, lr)
        res_hint = Results_many_runs(label='HintSGD', marker=markers[i])
        res_hint.read_logs(method, experiment, n_runs=len(seeds), eps=eps)
        res_hint.plot_train_loss(step=step, alpha=alpha, markevery=10, color=colors[i])

plt.yscale('log')
plt.legend()
plt.ylabel('Train loss')
if experiment == 'lenet':
    plt.yscale('linear')
# plt.xscale('log', base=10)
plt.xlabel('Total transmitted data (MB)')
if experiment == 'lenet':
    plt.title('LeNet')
elif experiment == 'resnet':
    plt.title('ResNet18')
elif experiment == 'resnet50':
    plt.title('ResNet50')

plt.tight_layout()
if not os.path.exists('./plots/'):
    os.makedirs('./plots/')
plot_name1 = '-'.join(algo_list)
plot_name2 = '-'.join(beta_list)
plot_name3 = '-'.join(sigma_sq_list)
plot_name4 = '-'.join(layerwise_list)
plot_name5 = '-'.join(alpha0_list)
plot_name6 = '-'.join(rr_list)
plt.savefig(
    './plots/train_loss_' + experiment + plot_name1 + '_' + plot_name2 + '_' + plot_name3 + '_' + plot_name4 + '_' + plot_name5 + '_' + plot_name6 + '.pdf',
    dpi=300, bbox_inches='tight')

f2 = plt.figure()
## Plot testing accuracy
for i, alg in enumerate(algo_list):
    beta = beta_list[i]
    sigma_sq = sigma_sq_list[i]
    layerwise = layerwise_list[i]
    random_round = rr_list[i]
    alpha0 = alpha0_list[i]
    if alg == 'IntSGD':
        method = '{0}_{1}_int_layerwise_{2}_rand_round_{3}_beta_{4}_sigma_sq_{5}_lr_{6}'.format(experiment, alpha0,
                                                                                                layerwise, random_round,
                                                                                                beta, sigma_sq, lr0)
        for (epoch, lr) in epoch_lrs:
            method += '_{}_{}'.format(epoch, lr)

        # int_label = 'IntSGD' + '(' + r'$\alpha_0=$' + str(
        #     alpha0) + r'$, \beta=$' + str(float(beta)) + r'$,\sigma_Q^2=$' + str(sigma_sq)
        int_label = 'IntSGD' + '(' + r'$\beta=$' + str(float(beta))

        if layerwise == 'True':
            int_label += ',lw'
        else:
            int_label += ''

        if random_round == 'True':
            int_label += ',rnd'
        else:
            int_label += ''

        int_label += ')'

        res_int = Results_many_runs(label=int_label, marker=markers[i])
        res_int.read_logs(method, experiment, n_runs=len(seeds), eps=eps)
        res_int.plot_test_acc(step=step, alpha=alpha, markevery=10, color=colors[i])
    elif alg == 'SGD':
        method = '{0}_{1}_sgd_lr_{2}'.format(experiment, str(batch_size), lr0)
        for (epoch, lr) in epoch_lrs:
            method += '_{}_{}'.format(epoch, lr)
        res_sgd = Results_many_runs(label='SGD', marker=markers[i])
        res_sgd.read_logs(method, experiment, n_runs=len(seeds), eps=eps)
        res_sgd.plot_test_acc(step=step, alpha=alpha, markevery=10, color=colors[i])
        plt.axhline(y=res_sgd.ave_test_acc[-1], color='black', linestyle='--')
    elif alg == 'NatSGD':
        method = '{0}_{1}_nat_sgd_lr_{2}'.format(experiment, str(batch_size), lr0)
        for (epoch, lr) in epoch_lrs:
            method += '_{}_{}'.format(epoch, lr)
        res_nat = Results_many_runs(label='NatSGD', marker=markers[i])
        res_nat.read_logs(method, experiment, n_runs=len(seeds), eps=eps)
        res_nat.plot_test_acc(step=step, alpha=alpha, markevery=10, color=colors[i])
    elif alg == 'HintSGD':
        method = '{0}_{1}_hint_sgd_lr_{2}'.format(experiment, str(batch_size), lr0)
        for (epoch, lr) in epoch_lrs:
            method += '_{}_{}'.format(epoch, lr)
        res_hint = Results_many_runs(label='HintSGD', marker=markers[i])
        res_hint.read_logs(method, experiment, n_runs=len(seeds), eps=eps)
        res_hint.plot_test_acc(step=step, alpha=alpha, markevery=10, color=colors[i])

# plt.yscale('log')
plt.legend()
plt.ylabel('Test Accuracy')
# plt.xscale('log', base=10)
if experiment == 'lenet':
    plt.ylim(60, 75)
# elif experiment == 'resnet':
else:
    plt.ylim(82, 95)
plt.xlabel('Total transmitted data (MB)')
if experiment == 'lenet':
    plt.title('LeNet')
elif experiment == 'resnet':
    plt.title('ResNet18')
elif experiment == 'resnet50':
    plt.title('ResNet50')

plt.tight_layout()
if not os.path.exists('./plots/'):
    os.makedirs('./plots/')
plot_name1 = '-'.join(algo_list)
plot_name2 = '-'.join(beta_list)
plot_name3 = '-'.join(sigma_sq_list)
plot_name4 = '-'.join(layerwise_list)
plot_name5 = '-'.join(alpha0_list)
plot_name6 = '-'.join(rr_list)
plt.savefig(
    './plots/test_acc_' + experiment + plot_name1 + '_' + plot_name2 + '_' + plot_name3 + '_' + plot_name4 + '_' + plot_name5 + '_' + plot_name6 + '.pdf',
    dpi=300, bbox_inches='tight')


f3 = plt.figure()
## Plot alpha
for i, alg in enumerate(algo_list):
    beta = beta_list[i]
    sigma_sq = sigma_sq_list[i]
    layerwise = layerwise_list[i]
    random_round = rr_list[i]
    alpha0 = alpha0_list[i]
    if alg == 'IntSGD':
        method = '{0}_{1}_int_layerwise_{2}_rand_round_{3}_beta_{4}_sigma_sq_{5}_lr_{6}'.format(experiment, alpha0,
                                                                                                layerwise, random_round,
                                                                                                beta, sigma_sq, lr0)
        for (epoch, lr) in epoch_lrs:
            method += '_{}_{}'.format(epoch, lr)

        # int_label = 'IntSGD' + '(' + r'$\alpha_0=$' + str(
        #     alpha0) + r'$, \beta=$' + str(float(beta)) + r'$,\sigma_Q^2=$' + str(sigma_sq)
        int_label = 'IntSGD' + '(' + r'$\beta=$' + str(float(beta))

        if layerwise == 'True':
            int_label += ',lw'
        else:
            int_label += ''

        if random_round == 'True':
            int_label += ',rnd'
        else:
            int_label += ''

        int_label += ')'

        res_int = Results_many_runs(label=int_label, marker=markers[i])
        res_int.read_logs(method, experiment, n_runs=len(seeds), eps=eps)
        res_int.plot_alphas(step=(100*step), alpha=alpha, markevery=10, color=colors[i])
    elif alg == 'SGD':
        continue
    elif alg == 'NatSGD':
        continue
    elif alg == 'HintSGD':
        method = '{0}_{1}_hint_sgd_lr_{2}'.format(experiment, str(batch_size), lr0)
        for (epoch, lr) in epoch_lrs:
            method += '_{}_{}'.format(epoch, lr)
        res_hint = Results_many_runs(label='HintSGD', marker=markers[i])
        res_hint.read_logs(method, experiment, n_runs=len(seeds), eps=eps)
        res_hint.plot_alphas(step=(100*step), alpha=alpha, markevery=10, color=colors[i])

# plt.yscale('log')
plt.legend()
plt.ylabel(r'$\alpha_k$')
plt.yscale('log', base=2)
plt.xlabel('Epochs')
if experiment == 'lenet':
    plt.title('LeNet')
elif experiment == 'resnet':
    plt.title('ResNet18')
elif experiment == 'resnet50':
    plt.title('ResNet50')

plt.tight_layout()
if not os.path.exists('./plots/'):
    os.makedirs('./plots/')
plot_name1 = '-'.join(algo_list)
plot_name2 = '-'.join(beta_list)
plot_name3 = '-'.join(sigma_sq_list)
plot_name4 = '-'.join(layerwise_list)
plot_name5 = '-'.join(alpha0_list)
plot_name6 = '-'.join(rr_list)
plt.savefig(
    './plots/alphas_' + experiment + plot_name1 + '_' + plot_name2 + '_' + plot_name3 + '_' + plot_name4 + '_' + plot_name5 + '_' + plot_name6 + '.pdf',
    dpi=300, bbox_inches='tight')


f4 = plt.figure()
## Plot max coords
for i, alg in enumerate(algo_list):
    beta = beta_list[i]
    sigma_sq = sigma_sq_list[i]
    layerwise = layerwise_list[i]
    random_round = rr_list[i]
    alpha0 = alpha0_list[i]
    if alg == 'IntSGD':
        method = '{0}_{1}_int_layerwise_{2}_rand_round_{3}_beta_{4}_sigma_sq_{5}_lr_{6}'.format(experiment, alpha0,
                                                                                                layerwise, random_round,
                                                                                                beta, sigma_sq, lr0)
        for (epoch, lr) in epoch_lrs:
            method += '_{}_{}'.format(epoch, lr)

        # int_label = 'IntSGD' + '(' + r'$\alpha_0=$' + str(
        #     alpha0) + r'$, \beta=$' + str(float(beta)) + r'$,\sigma_Q^2=$' + str(sigma_sq)
        int_label = 'IntSGD' + '(' + r'$\beta=$' + str(float(beta))

        if layerwise == 'True':
            int_label += ',lw'
        else:
            int_label += ''

        if random_round == 'True':
            int_label += ',rnd'
        else:
            int_label += ''

        int_label += ')'

        res_int = Results_many_runs(label=int_label, marker=markers[i])
        res_int.read_logs(method, experiment, n_runs=len(seeds), eps=eps)
        res_int.plot_max_coords(step=500, alpha=alpha, markevery=10, color=colors[i])
    elif alg == 'SGD':
        continue

# plt.yscale('log')
plt.legend()
plt.yscale('log', base=2)
plt.ylabel('Max integer to send')
if experiment == 'lenet':
    plt.title('Workers to master, LeNet')
elif experiment == 'resnet':
    plt.title('Workers to master, ResNet18')
elif experiment == 'resnet50':
    plt.title('Workers to master, ResNet50')
plt.xlabel('Epoch')
plt.tight_layout()
if not os.path.exists('./plots/'):
    os.makedirs('./plots/')
plot_name1 = '-'.join(algo_list)
plot_name2 = '-'.join(beta_list)
plot_name3 = '-'.join(sigma_sq_list)
plot_name4 = '-'.join(layerwise_list)
plot_name5 = '-'.join(alpha0_list)
plot_name6 = '-'.join(rr_list)
plt.savefig(
    './plots/max_coords_' + experiment + plot_name1 + '_' + plot_name2 + '_' + plot_name3 + '_' + plot_name4 + '_' + plot_name5 + '_' + plot_name6 + '.pdf',
    dpi=300, bbox_inches='tight')

# f5 = plt.figure()
# ## Plot train accuracy
# for i, alg in enumerate(algo_list):
#     beta = beta_list[i]
#     sigma_sq = sigma_sq_list[i]
#     layerwise = layerwise_list[i]
#     random_round = rr_list[i]
#     alpha0 = alpha0_list[i]
#     if alg == 'IntSGD':
#         method = '{0}_{1}_int_layerwise_{2}_rand_round_{3}_beta_{4}_sigma_sq_{5}_lr_{6}'.format(experiment, alpha0,
#                                                                                                 layerwise, random_round,
#                                                                                                 beta, sigma_sq, lr0)
#         for (epoch, lr) in epoch_lrs:
#             method += '_{}_{}'.format(epoch, lr)
#
#         int_label = 'IntSGD' + '(' + r'$\alpha_0=$' + str(
#             alpha0) + r'$, \beta=$' + str(float(beta)) + r'$,\sigma_Q^2=$' + str(sigma_sq)
#
#         if layerwise == 'True':
#             int_label += ',lw'
#         else:
#             int_label += ''
#
#         if random_round == 'True':
#             int_label += ',rnd'
#         else:
#             int_label += ''
#
#         int_label += ')'
#
#         res_int = Results_many_runs(label=int_label, marker=markers[i])
#         res_int.read_logs(method, experiment, n_runs=len(seeds), eps=eps)
#         res_int.plot_train_acc(step=step, alpha=alpha, markevery=10, color=colors[i])
#     elif alg == 'SGD':
#         method = '{0}_{1}_sgd_lr_{2}'.format(experiment, str(batch_size), lr0)
#         for (epoch, lr) in epoch_lrs:
#             method += '_{}_{}'.format(epoch, lr)
#         res_sgd = Results_many_runs(label='SGD', marker=markers[i])
#         res_sgd.read_logs(method, experiment, n_runs=len(seeds), eps=eps)
#         res_sgd.plot_train_acc(step=step, alpha=alpha, markevery=10, color=colors[i])
#
# # plt.yscale('log')
# plt.legend()
# if experiment == 'lenet':
#     plt.ylim([55, 75])
# elif experiment == 'resnet':
#     plt.ylim([82, 95])
# plt.ylabel('Train Accuracy')
# plt.xlabel('Epoch')
# if experiment == 'lenet':
#     plt.title('LeNet')
# elif experiment == 'resnet':
#     plt.title('ResNet18')
# plt.tight_layout()
# if not os.path.exists('./plots/'):
#     os.makedirs('./plots/')
# plot_name1 = '-'.join(algo_list)
# plot_name2 = '-'.join(beta_list)
# plot_name3 = '-'.join(sigma_sq_list)
# plot_name4 = '-'.join(layerwise_list)
# plot_name5 = '-'.join(alpha0_list)
# plot_name6 = '-'.join(rr_list)
# plt.savefig(
#     './plots/train_acc_' + experiment + plot_name1 + '_' + plot_name2 + '_' + plot_name3 + '_' + plot_name4 + '_' + plot_name5 + '_' + plot_name6 + '.pdf',
#     dpi=300)
#
# f5 = plt.figure()
# ## Plot max sum coords
# for i, alg in enumerate(algo_list):
#     beta = beta_list[i]
#     sigma_sq = sigma_sq_list[i]
#     layerwise = layerwise_list[i]
#     random_round = rr_list[i]
#     alpha0 = alpha0_list[i]
#     if alg == 'IntSGD':
#         method = '{0}_{1}_int_layerwise_{2}_rand_round_{3}_beta_{4}_sigma_sq_{5}_lr_{6}'.format(experiment, alpha0,
#                                                                                                 layerwise, random_round,
#                                                                                                 beta, sigma_sq, lr0)
#         for (epoch, lr) in epoch_lrs:
#             method += '_{}_{}'.format(epoch, lr)
#
#         int_label = 'IntSGD' + '(' + r'$\alpha_0=$' + str(
#             alpha0) + r'$, \beta=$' + str(float(beta)) + r'$,\sigma_Q^2=$' + str(sigma_sq)
#
#         if layerwise == 'True':
#             int_label += ',lw'
#         else:
#             int_label += ''
#
#         if random_round == 'True':
#             int_label += ',rnd'
#         else:
#             int_label += ''
#
#         int_label += ')'
#
#         res_int = Results_many_runs(label=int_label, marker=markers[i])
#         res_int.read_logs(method, experiment, n_runs=len(seeds), eps=eps)
#         res_int.plot_max_sum_coords(step=500, alpha=alpha, markevery=10, color=colors[i])
#     elif alg == 'SGD':
#         continue
#
# # plt.yscale('log')
# plt.legend()
# plt.yscale('log', base=2)
# plt.ylabel('Max integer to send')
# if experiment == 'lenet':
#     plt.title('Master to workers, LeNet')
# elif experiment == 'resnet':
#     plt.title('Master to workers, ResNet18')
# plt.xlabel('Epoch')
# plt.tight_layout()
# if not os.path.exists('./plots/'):
#     os.makedirs('./plots/')
# plot_name1 = '-'.join(algo_list)
# plot_name2 = '-'.join(beta_list)
# plot_name3 = '-'.join(sigma_sq_list)
# plot_name4 = '-'.join(layerwise_list)
# plot_name5 = '-'.join(alpha0_list)
# plot_name6 = '-'.join(rr_list)
# plt.savefig(
#     './plots/max_sum_coords_' + experiment + plot_name1 + '_' + plot_name2 + '_' + plot_name3 + '_' + plot_name4 + '_' + plot_name5 + '_' + plot_name6 + '.pdf',
#     dpi=300)
