'''
Preprocess and split the data, compute x_opt and f_opt
'''

import numpy as np
import argparse
import pickle
from datasets.utils import get_dataset
import numpy.linalg as la
from config import data_path, is_normalized, l2_weights, is_convex, shuffle
from loss_functions import LogisticRegression
from sklearn.preprocessing import normalize
import os

parser = argparse.ArgumentParser(description='Preprocessing for the dataset')
parser.add_argument('--data', action='store', dest='dataset', type=str, help='Which dataset?')
parser.add_argument('--cond', action='store', dest='cond', type=float, help='Condition of halting, grad norm sqr < ?')
parser.add_argument('--it_max', action='store', dest='it_max', type=int, help='Max iteration')
parser.add_argument('--n', action='store', dest='n_workers', type=int, help='Number of local workers')

args = parser.parse_args()
dataset = args.dataset
n_workers = args.n_workers
it_max = args.it_max
cond = args.cond

l2 = l2_weights[dataset]
if is_convex:
    l2 = 0

A, b = get_dataset(dataset, data_path)
print("===Data has been loaded===")
if is_normalized:
    A = normalize(A)

if shuffle:
    idx = np.arange(len(b))
    np.random.shuffle(idx)
    A = A[idx]
    b = b[idx]

N, d = A.shape
m = N // n_workers
A = A[:(m*n_workers)]
b = b[:(m*n_workers)]
N, _ = A.shape
x0 = np.zeros((d,))
loss_function = LogisticRegression(A, b, l2=l2)
L = loss_function.smoothness()


grad_norm_sq = la.norm(loss_function.gradient(x0))
x = np.copy(x0)
print('f_0: {0}'.format(loss_function.value(x)))

k = 0
while grad_norm_sq >= cond and k <= it_max:

    grad = loss_function.gradient(x)
    grad_norm_sq = loss_function.norm(grad) ** 2
    x = x - (1 / L) * grad

    if k % 50 == 0:
        print(grad_norm_sq)

    k += 1

x_opt = x.copy()
f_opt = loss_function.value(x_opt)

data_info = {'x_opt': x_opt, 'f_opt': f_opt, 'L': L}
if is_convex:
    cvx = 'cvx'
else:
    cvx = 'scvx'
dataset_name = '{0}-{1}-{2}'.format(dataset, cvx, n_workers)
data_set_path = '{0}{1}/'.format(data_path, dataset_name)
if not os.path.exists(data_set_path):
    os.makedirs(data_set_path)
file_name = '{0}{1}-{2}-{3}-{4}_info.p'.format(data_set_path, dataset, cond, it_max, l2)
pickle.dump(data_info, open(file_name, "wb"))

# Splitting the data
f_i = 0
for i in range(n_workers):
    Ai = A[(i * m):(i + 1) * m]
    bi = b[(i * m):(i + 1) * m]
    data_info = {'Ai': Ai, 'bi': bi}
    local_data_path = '{0}{1}/'.format(data_set_path, str(i))
    if not os.path.exists(local_data_path):
        os.makedirs(local_data_path)
    pickle.dump(Ai, open(local_data_path + 'A.p', "wb"))
    pickle.dump(bi, open(local_data_path + 'b.p', "wb"))

print("===Preprocessing is finished===")
