import numpy as onp
import os
from scipy.stats import bernoulli
import random


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    onp.random.seed(seed)

def rotate_func(l, n):
    return l[n:] + l[:n]

def int_quant(w):
    s = onp.sign(onp.array(w))
    w = onp.abs(w)
    lower = onp.floor(w)
    upper = onp.ceil(w)
    prob = (w - lower) / (upper - lower + onp.finfo(float).eps)
    prob[onp.isnan(prob)] = 0
    v = [bernoulli.rvs(p) for p in prob]
    return s * (lower + v)


def nat_compression(w):
    s = onp.sign(onp.array(w))
    w = onp.abs(w)
    power = onp.log2(w)
    lower = onp.power(2., onp.floor(power))
    upper = onp.power(2., onp.floor(power) + 1)
    prob = (w - lower) / (upper - lower + onp.finfo(float).eps)
    prob[onp.isnan(prob)] = 0
    v = [bernoulli.rvs(p) for p in prob]
    return s*lower*onp.power(2., v)

def run_gd(algos_path, n_workers, it_max, alpha_choice, beta, sigma_Q, dataset):
    file_name = algos_path + 'IntDCGD.py'
    os.system(
        "mpiexec -n {0} python {1} --it {2} --alpha {3} --beta {4} --sigma_Q {5} --data {6}".format(
            n_workers, file_name, it_max, alpha_choice, beta, sigma_Q, dataset
        ))
    print('#', end='')

def run_sgd(algos_path, n_workers, it_max, batch_size, alpha_choice, beta, sigma_Q, dataset):
    file_name = algos_path + 'IntDCSGD.py'
    os.system(
        "mpiexec -n {0} python {1} --it {2} --bs {3} --alpha {4} --beta {5} --sigma_Q {6} --data {7}".format(
            n_workers, file_name, it_max, batch_size, alpha_choice, beta, sigma_Q, dataset
        ))
    print('#', end='')


def run_diana(algos_path, n_workers, it_max, alpha_choice, beta, sigma_Q, dataset):
    file_name = algos_path + 'IntDIANA.py'
    os.system("mpiexec -n {0} python {1} --it {2} --alpha {3} --beta {4} --sigma_Q {5} --data {6}".format(
        n_workers, file_name, it_max, alpha_choice, beta, sigma_Q, dataset
    ))
    print('#', end='')


def run_vr_diana(algos_path, n_workers, it_max, batch_size, alpha_choice, beta, sigma_Q, dataset, p):
    file_name = algos_path + 'VR-IntDIANA.py'
    os.system(
        "mpiexec -n {0} python {1} --it {2}  --bs {3} --alpha {4} --beta {5} --sigma_Q {6} --data {7} --p {8}".format(
            n_workers, file_name, it_max, batch_size, alpha_choice, beta, sigma_Q, dataset, p
        ))
    print('#', end='')

def run_vr_hintdiana(algos_path, n_workers, it_max, batch_size, alpha_choice, beta, sigma_Q, dataset, p):
    file_name = algos_path + 'VR-HintDIANA.py'
    os.system(
        "mpiexec -n {0} python {1} --it {2}  --bs {3} --alpha {4} --beta {5} --sigma_Q {6} --data {7} --p {8}".format(
            n_workers, file_name, it_max, batch_size, alpha_choice, beta, sigma_Q, dataset, p
        ))
    print('#', end='')

def run_vr_natdiana(algos_path, n_workers, it_max, batch_size, alpha_choice, beta, sigma_Q, dataset, p):
    file_name = algos_path + 'VR-NatDIANA.py'
    os.system(
        "mpiexec -n {0} python {1} --it {2}  --bs {3} --alpha {4} --beta {5} --sigma_Q {6} --data {7} --p {8}".format(
            n_workers, file_name, it_max, batch_size, alpha_choice, beta, sigma_Q, dataset, p
        ))
    print('#', end='')

def run_vr(algos_path, n_workers, it_max, batch_size, alpha_choice, beta, sigma_Q, dataset, p):
    file_name = algos_path + 'L-SVRG.py'
    os.system(
        "mpiexec -n {0} python {1} --it {2}  --bs {3} --alpha {4} --beta {5} --sigma_Q {6} --data {7} --p {8}".format(
            n_workers, file_name, it_max, batch_size, alpha_choice, beta, sigma_Q, dataset, p
        ))
    print('#', end='')
