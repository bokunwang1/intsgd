from config import data_info, algos_path, default_it_max
from utils import run_vr_natdiana, run_vr_diana, run_vr_hintdiana, run_vr
import argparse

parser = argparse.ArgumentParser(description='Run the Algorithms')
# parser.add_argument('--it', action='store', dest='it_max', type=int, help='Numer of Iterations')
parser.add_argument('--alpha', action='store', dest='alpha_choice', default='adaptive', type=str, help='Type of alpha')
parser.add_argument('--beta', action='store', dest='beta', default=0.0, type=float, help='Value of beta')
parser.add_argument('--sigma_Q', action='store', dest='sigma_Q', default=0.001, type=float, help='Value of sigma_Q')
parser.add_argument('--data', action='store', dest='dataset', default='a5a', type=str, help='Dataset')
parser.add_argument('--alg', action='store', dest='algo_name', type=str,
                    help='Which algorithm: IntDCGD, IntDCSGD, IntDIANA, VR-IntDIANA')
parser.add_argument('--n', action='store', dest='n_workers', type=int, help='Number of workers')
# parser.add_argument('--bs', action='store', dest='batch_size', type=int, help='Batch size for stochastic algorithms')

args = parser.parse_args()
# it_max = args.it_max
alpha_choice = args.alpha_choice
beta = args.beta
sigma_Q = args.sigma_Q
dataset = args.dataset
algo_name = args.algo_name
n_workers = args.n_workers
# batch_size = args.batch_size

N, d = data_info[dataset]

if algo_name == 'VR-IntDIANA':
    batch_size = N // (20 * n_workers)
    it_max = int(default_it_max * 20)
    p = batch_size / (N / n_workers)
    run_vr_diana(algos_path, n_workers, it_max, batch_size, alpha_choice, beta, sigma_Q, dataset, p)
elif algo_name == 'VR-HintDIANA':
    batch_size = N // (20 * n_workers)
    it_max = int(default_it_max * 20)
    p = batch_size / (N / n_workers)
    run_vr_hintdiana(algos_path, n_workers, it_max, batch_size, alpha_choice, beta, sigma_Q, dataset, p)
elif algo_name == 'VR-NatDIANA':
    batch_size = N // (20 * n_workers)
    it_max = int(default_it_max * 20)
    p = batch_size / (N / n_workers)
    run_vr_natdiana(algos_path, n_workers, it_max, batch_size, alpha_choice, beta, sigma_Q, dataset, p)
elif algo_name == 'L-SVRG':
    batch_size = N // (20 * n_workers)
    it_max = int(default_it_max * 20)
    p = batch_size / (N / n_workers)
    run_vr(algos_path, n_workers, it_max, batch_size, alpha_choice, beta, sigma_Q, dataset, p)
else:
    raise ValueError('The algorithm has not been implemented!')
