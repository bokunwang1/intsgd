import os, sys
from pathlib import Path  # if you haven't already done so

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass
from loss_functions.logistic_regression import LogisticRegression
from helpers.lr_handler import LR_Handler
from helpers.alpha_handler import Alpha_Handler
from helpers.trace_handler import Trace_Handler
from scipy.stats import bernoulli
from utils import set_seed, int_quant, nat_compression
import numpy as onp
import argparse
from mpi4py import MPI
from config import data_path, eval_ratio, l2_weights, lr_decay, seeds, is_convex
import pickle
import glob

parser = argparse.ArgumentParser(description='Run DCSGD')
parser.add_argument('--it', action='store', dest='it_max', type=int, help='Numer of Iterations')
parser.add_argument('--bs', action='store', dest='batch_size', type=int, help='Batch size')
parser.add_argument('--alpha', action='store', dest='alpha_choice', type=str, help='Type of alpha')
parser.add_argument('--beta', action='store', dest='beta', type=float, help='Value of beta')
parser.add_argument('--sigma_Q', action='store', dest='sigma_Q', type=float, help='Value of sigma_Q')
parser.add_argument('--data', action='store', dest='dataset', type=str, help='Dataset')
parser.add_argument('--p', action='store', dest='prob', type=float, help='Probability for SVRG')

args = parser.parse_args()
it_max = args.it_max
batch_size = args.batch_size
alpha_choice = args.alpha_choice
beta = args.beta
sigma_Q = args.sigma_Q
dataset = args.dataset
prob = args.prob

if is_convex == True:
    l2 = 0
    problem_type = 'cvx'
else:
    l2 = l2_weights[dataset]
    problem_type = 'scvx'

trace_period = onp.floor(it_max * eval_ratio)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_workers = comm.Get_size()

# Load local data
file_name = '{0}{1}-{2}-{3}/{4}'.format(data_path, dataset, problem_type, n_workers, rank)
with open(file_name + '/A.p', "rb") as file:
    Ai = pickle.load(file)
with open(file_name + '/b.p', "rb") as file:
    bi = pickle.load(file)

m = bi.shape[0]
loss_i = LogisticRegression(Ai, bi, l2=l2)

# initial point
x0 = onp.zeros((Ai.shape[1],))
h0 = onp.zeros((Ai.shape[1],))
fi_0 = loss_i.value(x0)
f_0 = comm.allreduce(fi_0, op=MPI.SUM) / n_workers

# load opt sol for eval
if rank == 0:
    opt_file_path = '{0}{1}-{2}-{3}'.format(data_path, dataset, problem_type, n_workers)
    abs_path = glob.glob(opt_file_path + '/*_info.p')
    with open(abs_path[0], "rb") as file:
        opt_info = pickle.load(file)

    x_opt = opt_info['x_opt']
    f_opt = opt_info['f_opt']
    L = opt_info['L']
    # if problem_type == 'cvx':
    #     r_0 = f_0 - f_opt
    # else:
    #     r_0 = onp.linalg.norm(x0 - x_opt) ** 2

    trace_handler = Trace_Handler(all_seeds=seeds[0], problem_type=problem_type, x_opt=x_opt, f_opt=f_opt)
else:
    L = None
    trace_handler = None
    # r_0 = None

L = comm.bcast(L, root=0)
Li_batch = loss_i.batch_smoothness(batch_size=batch_size)
max_Li_batch = comm.allreduce(Li_batch, op=MPI.MAX)

if problem_type == 'cvx':
    lr0 = 1 / (L * (1 + 36 * (1 + 1 / 8) / n_workers))
else:
    lr0 = 1 / (L * (1 + 36 * (1 + 1 / 8) / n_workers))

if rank == 0:
    lr_handler = LR_Handler(num_iters=it_max, decay=lr_decay, lr0=lr0)
    alpha0 = (onp.sqrt(Ai.shape[1]) / sigma_Q) * onp.ones_like(x0)
    alpha_handler = Alpha_Handler(alpha_choice=alpha_choice, lr=lr_handler,
                                  sigma_Q=sigma_Q, num_workers=n_workers, beta=beta,
                                  alpha0=alpha0)
    all_mbs = {}

else:
    lr_handler = None
    alpha_handler = None

seeds_i = seeds[rank]

for seed in seeds_i:
    # set the random seed
    # seed_i = seed  # + rank  # different seeds for different workers
    set_seed(seed)
    x = onp.copy(x0)
    hi_k = onp.copy(h0)
    h_k = onp.copy(h0)
    w_i = onp.copy(x0)
    u_i = loss_i.gradient(w_i)
    if rank == 0:
        lr_handler.set_lr()

    n_bits = 0
    n_grad = 0
    for k in range(it_max):
        if rank == 0:
            lr = lr_handler.get_lr()
        else:
            lr = None

        lr = comm.bcast(lr, root=0)
        idx = onp.random.choice(bi.shape[0], size=batch_size)
        local_g_x = loss_i.stochastic_gradient(x, idx)
        local_g_w = loss_i.stochastic_gradient(w_i, idx)
        local_sgrad = local_g_x - local_g_w + u_i
        if k == 0:
            Qi_k = local_sgrad - hi_k
            n_bit = 0
        else:
            Qi_k = nat_compression(local_sgrad - hi_k)
            max_abs_i = onp.max(onp.abs(Qi_k))
            n_bit = 9

        n_bit_all = comm.allreduce(n_bit, op=MPI.SUM)
        n_bits += n_bit_all

        hi_k_next = hi_k + Qi_k / (1 + 1 / 8)
        hi_k = onp.copy(hi_k_next)

        Q_k = comm.allreduce(Qi_k, op=MPI.SUM)

        sum_n_bit = 32

        sum_n_bit_all = comm.allreduce(sum_n_bit, op=MPI.SUM)
        n_bits += sum_n_bit_all

        if k % trace_period == 0:
            max_val_i = onp.max(onp.abs(Qi_k))
            max_val = comm.allreduce(max_val_i, op=MPI.MAX)
            max_val_sum = onp.max(onp.abs(Q_k))

            fi_k = loss_i.value(x)
            f_k = comm.allreduce(fi_k, op=MPI.SUM) / n_workers

            if rank == 0:
                trace_handler.update_trace(seed=seed, x=x, f_eval=f_k)
                trace_handler.update_max_value(seed=seed, max_val=max_val, max_val_sum=max_val_sum, alpha=None)
                trace_handler.update_ngrads(n_grads=n_workers * trace_period, seed=seed, iter=k)
                trace_handler.update_mbs(n_bits=n_bits, seed=seed)

        nu_k = bernoulli.rvs(prob)
        nu_k = comm.bcast(nu_k, root=0)

        if nu_k == 1:
            w_i_next = x
            u_i = loss_i.gradient(w_i_next)
            w_i = onp.copy(w_i_next)
            n_grad += m
        else:
            n_grad += batch_size

        if k % trace_period == 0:
            n_grad_all = comm.allreduce(n_grad, op=MPI.SUM)
            if rank == 0:
                trace_handler.update_ngrads(n_grads=n_grad_all, seed=seed, iter=k)
            n_grad = 0
            n_grad_all = 0

        g_k = h_k + Q_k / n_workers
        x_prev = onp.copy(x)
        x = x - lr * g_k

        h_k_next = h_k + Q_k / ((1 + 1 / 8) * n_workers)
        h_k = onp.copy(h_k_next)

print("Rank %d is down" % rank)

if rank == 0:
    grad_oracles_trace, dist_trace, fgap_trace, max_value_trace, alpha_trace, iter_trace, mbs_trace = trace_handler.get_trace()
    algo_name = 'VR-NatDIANA'
    save_path = '{0}/{1}-{2}-{3}-{4}-{5}-{6}'.format('./results', dataset, problem_type, alpha_choice, beta, algo_name,
                                                     n_workers)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pickle.dump(grad_oracles_trace, open(save_path + "/grad_oracles_trace.p", "wb"))
    pickle.dump(dist_trace, open(save_path + "/dist_trace.p", "wb"))
    pickle.dump(fgap_trace, open(save_path + "/fgap_trace.p", "wb"))
    pickle.dump(max_value_trace, open(save_path + "/max_value_trace.p", "wb"))
    pickle.dump(iter_trace, open(save_path + "/iter_trace.p", "wb"))
    pickle.dump(mbs_trace, open(save_path + "/mbs_trace.p", "wb"))
