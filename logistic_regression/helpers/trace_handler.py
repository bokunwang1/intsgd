import numpy as np


class Trace_Handler(object):
    def __init__(self, all_seeds, problem_type, x_opt, f_opt):
        super().__init__()
        self.problem_type = problem_type
        self.all_seeds = all_seeds
        self.x_opt = x_opt
        self.f_opt = f_opt
        self.grad_oracles = {}
        self.dist_trace = {}
        self.mbs_trace = {}
        self.fgap_trace = {}
        self.max_value = {}
        self.time = {}
        self.alpha_trace = {}
        self.iter_trace = {}
        for seed in self.all_seeds:
            self.grad_oracles[seed] = []
            self.max_value[seed] = []
            self.time[seed] = []
            self.alpha_trace[seed] = []
            self.iter_trace[seed] = []
            self.dist_trace[seed] = []
            self.fgap_trace[seed] = []
            self.mbs_trace[seed] = []

    def get_trace(self):
        return self.grad_oracles, self.dist_trace, self.fgap_trace, self.max_value, self.alpha_trace, self.iter_trace, self.mbs_trace

    def update_ngrads(self, n_grads, seed, iter):
        if iter == 0:
            cur_grads = 0
        else:
            cur_grads = self.grad_oracles[seed][-1]
            cur_grads += n_grads
        self.grad_oracles[seed].append(cur_grads)
        self.iter_trace[seed].append(iter)

    def update_max_value(self, seed, max_val, max_val_sum, alpha):
        val = np.maximum(max_val, max_val_sum)
        self.max_value[seed].append(max_val)
        # self.max_value_sum[seed].append(max_val_sum)
        self.alpha_trace[seed].append(alpha)

    def update_mbs(self, seed, n_bits):
        self.mbs_trace[seed].append(n_bits / (8 * 1e6))

    def update_trace(self, seed, x, f_eval=np.inf):
        # if self.problem_type == 'scvx':
        # distance = np.linalg.norm(x - self.x_opt)**2/r_0

        distance = np.linalg.norm(x - self.x_opt) ** 2
        self.dist_trace[seed].append(distance.item())
        # elif self.problem_type == 'cvx':
        # subopt_gap = (f_eval - self.f_opt)/r_0
        subopt_gap = f_eval - self.f_opt
        self.fgap_trace[seed].append(subopt_gap)

        # else:
        # raise ValueError('Not implemented!')
