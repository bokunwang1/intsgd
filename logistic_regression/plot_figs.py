from config import data_info, plots_path, is_convex
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as onp
import pickle
import argparse
import os

sns.set(style="whitegrid", context="talk", font_scale=1.2, palette=sns.color_palette("bright"), color_codes=False)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.figsize'] = (8, 6)

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['mathtext.fontset'] = 'cm'

parser = argparse.ArgumentParser(description='Plotting the figures')
parser.add_argument('--algs', nargs='+', dest='algo_list', help='List of algorithms')
parser.add_argument('--alphas', nargs='+', dest='alpha_list', help='List of choices of alpha')
parser.add_argument('--betas', nargs='+', dest='betas', help='List of values of beta')
parser.add_argument('--data', action='store', dest='dataset', type=str, help='Dataset')
parser.add_argument('--n', action='store', dest='n_workers', type=int, help='Number of workers')

args = parser.parse_args()
algo_list = args.algo_list
betas = args.betas
n_workers = args.n_workers
dataset = args.dataset
alpha_list = args.alpha_list
N, d = data_info[dataset]

if is_convex == True:
    problem_type = 'cvx'
else:
    problem_type = 'scvx'

if len(algo_list) != len(alpha_list) or len(alpha_list) != len(betas):
    raise ValueError('The length of algorithm list should be equal to alpha list! They are one-one corresponded.')

all_dist_trace_ave = {}
all_lower_dist_trace = {}
all_upper_dist_trace = {}
all_fgap_trace_ave = {}
all_lower_fgap_trace = {}
all_upper_fgap_trace = {}
all_max_val_trace_ave = {}
all_lower_max_val_trace = {}
all_upper_max_val_trace = {}
all_iter_trace_ave = {}
all_grad_oracles_trace_ave = {}
all_mbs_ave = {}

for alg, alpha_choice, beta in zip(algo_list, alpha_list, betas):
    save_path = '{0}/{1}-{2}-{3}-{4}-{5}-{6}'.format('./results', dataset, problem_type, alpha_choice, beta, alg,
                                                     str(n_workers))
    with open(save_path + '/dist_trace.p', "rb") as file:
        dist_trace = pickle.load(file)
    with open(save_path + '/fgap_trace.p', "rb") as file:
        fgap_trace = pickle.load(file)
    with open(save_path + '/max_value_trace.p', "rb") as file:
        max_val_trace = pickle.load(file)
    with open(save_path + '/iter_trace.p', "rb") as file:
        iter_trace = pickle.load(file)
    with open(save_path + '/grad_oracles_trace.p', "rb") as file:
        grad_oracles_trace = pickle.load(file)
    with open(save_path + '/mbs_trace.p', "rb") as file:
        mbs_trace = pickle.load(file)

    dist_trace_lst = list(dist_trace.values())
    fgap_trace_lst = list(fgap_trace.values())

    mbs_trace_lst = list(mbs_trace.values())
    mbs_trace_ave = onp.mean(mbs_trace_lst, axis=0)

    dist_trace_log = [onp.log(y) for y in dist_trace_lst]
    dist_trace_log_ave = onp.mean(dist_trace_log, axis=0)
    dist_trace_log_std = onp.std(dist_trace_log, axis=0)
    dist_trace_ave = onp.exp(dist_trace_log_ave)
    lower_dist_trace, upper_dist_trace = onp.exp(dist_trace_log_ave - dist_trace_log_std), onp.exp(
        dist_trace_log_ave + dist_trace_log_std)

    fgap_trace_log = [onp.log(y) for y in fgap_trace_lst]
    fgap_trace_log_ave = onp.mean(fgap_trace_log, axis=0)
    fgap_trace_log_std = onp.std(fgap_trace_log, axis=0)
    fgap_trace_ave = onp.exp(fgap_trace_log_ave)
    lower_fgap_trace, upper_fgap_trace = onp.exp(fgap_trace_log_ave - fgap_trace_log_std), onp.exp(
        fgap_trace_log_ave + fgap_trace_log_std)

    if alg == 'IntDCGD':
        alg = 'IntGD'
    elif alg == 'IntDCSGD':
        alg = 'IntSGD'

    all_dist_trace_ave[alg + alpha_choice + str(beta)] = dist_trace_ave
    all_lower_dist_trace[alg + alpha_choice + str(beta)] = lower_dist_trace
    all_upper_dist_trace[alg + alpha_choice + str(beta)] = upper_dist_trace

    all_fgap_trace_ave[alg + alpha_choice + str(beta)] = fgap_trace_ave
    all_lower_fgap_trace[alg + alpha_choice + str(beta)] = lower_fgap_trace
    all_upper_fgap_trace[alg + alpha_choice + str(beta)] = upper_fgap_trace

    max_val_trace_lst = list(max_val_trace.values())
    max_val_trace_ave = onp.mean(max_val_trace_lst, axis=0)
    max_val_trace_std = onp.std(max_val_trace_lst, axis=0)

    lower_max_val_trace, upper_max_val_trace = (max_val_trace_ave - max_val_trace_std), (
            max_val_trace_ave + max_val_trace_std)

    all_max_val_trace_ave[alg + alpha_choice + str(beta)] = max_val_trace_ave
    all_lower_max_val_trace[alg + alpha_choice + str(beta)] = lower_max_val_trace
    all_upper_max_val_trace[alg + alpha_choice + str(beta)] = upper_max_val_trace

    iter_trace_ave = onp.average(list(iter_trace.values()), axis=0)
    grad_oracles_trace_ave = onp.average(list(grad_oracles_trace.values()), axis=0)
    all_iter_trace_ave[alg + alpha_choice + str(beta)] = iter_trace_ave
    all_grad_oracles_trace_ave[alg + alpha_choice + str(beta)] = grad_oracles_trace_ave / N
    all_mbs_ave[alg + alpha_choice + str(beta)] = mbs_trace_ave


f1 = plt.figure()
colors = ['r', 'y', 'm', 'b', 'g', 'c']
markers = ['D', 'o', 'x', '*', 'v', '.']
vis_time = 10

for i, alg in enumerate(algo_list):
    if alg == 'IntDCGD':
        alg = 'IntGD'
    elif alg == 'IntDCSGD':
        alg = 'IntSGD'

    alpha_choice = alpha_list[i]
    beta = float(betas[i])
    grad_oracles_trace_ave = all_grad_oracles_trace_ave[alg + alpha_choice + str(beta)]

    dist_trace_ave = all_dist_trace_ave[alg + alpha_choice + str(beta)]
    lower_dist_trace = all_lower_dist_trace[alg + alpha_choice + str(beta)]
    upper_dist_trace = all_upper_dist_trace[alg + alpha_choice + str(beta)]
    mbs_trace = all_mbs_ave[alg + alpha_choice + str(beta)]

    if alpha_list[i] == 'adaptive':
        alpha_name = 'ad'
    elif alpha_list[i] == 'constant':
        alpha_name = 'cs'
    elif alpha_list[i] == 'moving_avg':
        alpha_name = 'mavg'
    elif alpha_list[i] == 'moving_avg_w_const':
        alpha_name = 'sg'
    else:
        raise ValueError('Unkown type of alpha')

    if alpha_name == 'mavg':
        legend_name = alg + '-' + alpha_name + '(' + r'$\beta=$' + str(float(beta)) + ')'
    elif alpha_name == 'sg':
        legend_name = alg + '-' + alpha_name + '(' + r'$\beta=$' + str(float(beta)) + r'$,\sigma_Q=$' + str(0.001) + ')'
    else:
        legend_name = alg
    plot = plt.plot(mbs_trace[:len(dist_trace_ave)], dist_trace_ave, markevery=max(1, len(grad_oracles_trace_ave) // vis_time),
                    marker=markers[i], label=legend_name, color=colors[i])
    plt.fill_between(mbs_trace[:len(dist_trace_ave)], lower_dist_trace, upper_dist_trace, alpha=0.25, color=colors[i])

plt.yscale('log')
plt.tight_layout()
plt.legend()
plt.title('{0}, n={1}'.format(dataset, str(n_workers)))
plt.ylabel(r'$\Vert x-x^*\Vert^2$')
plt.xlabel('Transmitted Data (MB)')
# plt.xlim(right=max(all_grad_oracles_trace_ave['IntGD' + alpha_list[1] + betas[1]]))
fig_path = '{0}/{1}-{2}-{3}'.format(plots_path, dataset, problem_type, str(n_workers))
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

plot_name1 = '-'.join(algo_list)
plot_name2 = '-'.join(alpha_list)
plot_name3 = '-'.join(betas)
plt.savefig(fig_path + '/' + plot_name1 + '_' + plot_name2 + '_' + plot_name3 + '_dist_trace.pdf', dpi=300,
            bbox_inches='tight')

f2 = plt.figure()
vis_time = 10

for i, alg in enumerate(algo_list):

    if alg == 'IntDCGD':
        alg = 'IntGD'
    elif alg == 'IntDCSGD':
        alg = 'IntSGD'

    alpha_choice = alpha_list[i]
    beta = float(betas[i])
    grad_oracles_trace_ave = all_grad_oracles_trace_ave[alg + alpha_choice + str(beta)]

    fgap_trace_ave = all_fgap_trace_ave[alg + alpha_choice + str(beta)]
    lower_fgap_trace = all_lower_fgap_trace[alg + alpha_choice + str(beta)]
    upper_fgap_trace = all_upper_fgap_trace[alg + alpha_choice + str(beta)]
    mbs_trace = all_mbs_ave[alg + alpha_choice + str(beta)]
    if alpha_list[i] == 'adaptive':
        alpha_name = 'ad'
    elif alpha_list[i] == 'constant':
        alpha_name = 'cs'
    elif alpha_list[i] == 'moving_avg':
        alpha_name = 'mavg'
    elif alpha_list[i] == 'moving_avg_w_const':
        alpha_name = 'sg'
    else:
        raise ValueError('Unkown type of alpha')
    if alpha_name == 'mavg':
        legend_name = alg + '-' + alpha_name + '(' + r'$\beta=$' + str(float(beta)) + ')'
    elif alpha_name == 'sg':
        legend_name = alg + '-' + alpha_name + '(' + r'$\beta=$' + str(float(beta)) + r'$,\sigma_Q=$' + str(0.001) + ')'
    else:
        legend_name = alg
    plot = plt.plot(mbs_trace[:len(fgap_trace_ave)], fgap_trace_ave, markevery=max(1, len(grad_oracles_trace_ave) // vis_time),
                    marker=markers[i], label=legend_name, color=colors[i])
    plt.fill_between(mbs_trace[:len(fgap_trace_ave)], lower_fgap_trace, upper_fgap_trace, alpha=0.25, color=colors[i])

plt.yscale('log')
plt.tight_layout()
plt.legend()
plt.title('{0}, n={1}'.format(dataset, str(n_workers)))
plt.ylabel(r'$f(x) - f(x^*)$')
plt.xlabel('Transmitted Data (MB)')
# plt.xlim(right=max(all_grad_oracles_trace_ave['IntGD' + alpha_list[1] + betas[1]]))
fig_path = '{0}/{1}-{2}-{3}'.format(plots_path, dataset, problem_type, str(n_workers))
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

plot_name1 = '-'.join(algo_list)
plot_name2 = '-'.join(alpha_list)
plot_name3 = '-'.join(betas)
plt.savefig(fig_path + '/' + plot_name1 + '_' + plot_name2 + '_' + plot_name3 + '_fgap_trace.pdf', dpi=300,
            bbox_inches='tight')
#
# f3 = plt.figure()
# vis_time = 20
#
# for i, alg in enumerate(algo_list):
#     if alg == 'IntDCGD':
#         alg = 'IntGD'
#     elif alg == 'IntDCSGD':
#         alg = 'IntSGD'
#     alpha_choice = alpha_list[i]
#     beta = float(betas[i])
#     grad_oracles_trace_ave = all_grad_oracles_trace_ave[alg + alpha_choice + str(beta)]
#     max_val_trace_ave = all_max_val_trace_ave[alg + alpha_choice + str(beta)]
#     lower_max_val_trace = all_lower_max_val_trace[alg + alpha_choice + str(beta)]
#     upper_max_val_trace = all_upper_max_val_trace[alg + alpha_choice + str(beta)]
#     if alpha_list[i] == 'adaptive':
#         alpha_name = 'ad'
#     elif alpha_list[i] == 'constant':
#         alpha_name = 'cs'
#     elif alpha_list[i] == 'moving_avg':
#         alpha_name = 'mavg'
#     elif alpha_list[i] == 'moving_avg_w_const':
#         alpha_name = 'sg'
#     else:
#         raise ValueError('Unkown type of alpha')
#     if alpha_name == 'mavg':
#         legend_name = alg + '-' + alpha_name + '(' + r'$\beta=$' + str(float(beta)) + ')'
#     elif alpha_name == 'sg':
#         legend_name = alg + '-' + alpha_name + '(' + r'$\beta=$' + str(float(beta)) + r'$,\sigma_Q=$' + str(0.001) + ')'
#     else:
#         legend_name = alg + '-' + alpha_name
#     plot = plt.plot(grad_oracles_trace_ave[1:len(max_val_trace_ave)], max_val_trace_ave[1:],
#                     markevery=max(1, len(grad_oracles_trace_ave) // vis_time),
#                     marker=markers[i], label=legend_name, color=colors[i])
#     plt.fill_between(grad_oracles_trace_ave[1:len(max_val_trace_ave)], lower_max_val_trace[1:], upper_max_val_trace[1:], alpha=0.25,
#                      color=colors[i])
#
# plt.yscale('log', base=2)
# plt.tight_layout()
# plt.legend(prop={'size': 12})
# plt.title('{0}, n={1}'.format(dataset, str(n_workers)))
# plt.ylabel('Max integer to send')
# # plt.xlim([0, max(all_grad_oracles_trace_ave['IntGD' + alpha_list[1] + betas[1]])])
# plt.xlabel('#grad/mn')
# plt.ylim(bottom=1)
# fig_path = '{0}/{1}-{2}-{3}'.format(plots_path, dataset, problem_type, str(n_workers))
# if not os.path.exists(fig_path):
#     os.makedirs(fig_path)
#
# plot_name1 = '-'.join(algo_list)
# plot_name2 = '-'.join(alpha_list)
# plot_name3 = '-'.join(betas)
# plt.savefig(fig_path + '/' + plot_name1 + '_' + plot_name2 + '_' + plot_name3 + '_max_val.pdf', dpi=300,
#             bbox_inches='tight')
