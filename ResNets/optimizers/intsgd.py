import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np
from utils import int_quant

class IntSGD(Optimizer):
    r"""Implements stochastic gradient descent with integer quantization
    """

    def __init__(self, params, total_dim, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, n_fake_workers=8,
                 layerwise=False, coordwise=False, sigma_sq=1e-8,
                 alpha_coef=1., random_round=False, beta=0):

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if momentum > 0.0:
            print("No theory is available for positive momentum, so there can be issues")
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(IntSGD, self).__init__(params, defaults)

        if coordwise:
            raise ValueError("Coordwise is not supported at the moment")

        self.i_worker = 0
        self.n_fake_workers = n_fake_workers
        self.layerwise = layerwise
        self.coordwise = coordwise
        self.sigma_sq = sigma_sq
        self.alpha_coef = alpha_coef
        #####################
        self.total_dim = total_dim
        if layerwise:
            n_params = len(self.param_groups[0]['params'])
            self.all_alpha = [None] * n_params
            self.r = [0] * n_params
        else:
            self.r = 0
            self.alpha = None
            self.alphas = []
        #####################
        self.random_round = random_round
        self.beta = beta
        self.max_coords = []
        self.max_coord = 0
        self.max_sum_coords = []
        self.max_sum_coord = 0
        self.n_mbs = []
        self.n_bit = 0
        # self.n_bits_sgd = []
        # self.n_bit_sgd = 0

    def __setstate__(self, state):
        super(IntSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if len(self.param_groups) > 1:
            raise ValueError('Only 1 param group is supported now!')

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            update_norm_sq = 0

            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad
                dim_i = np.prod(list(d_p.shape))
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                param_state = self.state[p]
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                if 'parallel_buffer' not in param_state:
                    par_buf = param_state['parallel_buffer'] = torch.clone(d_p).detach() / self.n_fake_workers
                else:
                    par_buf = param_state['parallel_buffer']
                    if self.layerwise:
                        alpha_i = self.all_alpha[i]
                        if alpha_i is not None:
                            if self.random_round:
                                int_update = int_quant(alpha_i * torch.clone(d_p).detach())
                            else:
                                int_update = torch.round(alpha_i * torch.clone(d_p).detach())
                            # nnz = int_update[torch.abs(int_update) >= 1]
                            # self.n_bit += torch.sum(torch.ceil(torch.log2(torch.abs(nnz)))).item()
                            max_abs_i = int_update.norm(float('inf')).item()
                            if max_abs_i < 1:
                                self.n_bit += int_update.numel()
                            else:
                                self.n_bit += int_update.numel() * (1 + np.ceil(np.log2(max_abs_i)))

                            self.max_coord = max(self.max_coord, int_update.norm(float('inf')).item())
                            par_buf.add_(int_update, alpha=1 / self.n_fake_workers / alpha_i)
                        else:
                            "The first iteration is without quantization"
                            par_buf.add_(torch.clone(d_p).detach(), alpha=1 / self.n_fake_workers)
                            self.n_bit += torch.clone(d_p).detach().numel() * 32 * self.n_fake_workers
                    else:
                        if self.alpha is not None:
                            if self.random_round:
                                int_update = int_quant(self.alpha * torch.clone(d_p).detach())
                            else:
                                int_update = torch.round(self.alpha * torch.clone(d_p).detach())
                            self.max_coord = max(self.max_coord, int_update.norm(float('inf')).item())

                            max_abs_i = torch.abs(int_update.norm(float('inf'))).item()
                            if max_abs_i < 1:
                                self.n_bit += int_update.numel()
                            else:
                                self.n_bit += int_update.numel() * (1 + np.ceil(np.log2(max_abs_i)))

                            par_buf.add_(int_update, alpha=1 / self.n_fake_workers / self.alpha)
                        else:
                            "The first iteration is without quantization"
                            par_buf.add_(torch.clone(d_p).detach(), alpha=1 / self.n_fake_workers)
                            self.n_bit += torch.clone(d_p).detach().numel() * 32 * self.n_fake_workers

                if self.i_worker % self.n_fake_workers == self.n_fake_workers - 1:
                    "The parameters are only updated when each fake worker computed and quantized its updated"
                    p.add_(par_buf, alpha=-group['lr'])

                    if not self.layerwise:
                        if self.alpha is not None:
                            q_sum = par_buf * self.n_fake_workers * self.alpha
                            max_abs_sum = q_sum.norm(float('inf')).item()
                            if max_abs_sum < 1:
                                self.n_bit += par_buf.numel() * self.n_fake_workers
                            else:
                                self.n_bit += par_buf.numel() * (1 + np.ceil(np.log2(max_abs_sum))) * self.n_fake_workers
                        else:
                            self.n_bit += par_buf.numel() * 32 * self.n_fake_workers
                    else:
                        if alpha_i is not None:
                            q_sum = par_buf * self.n_fake_workers * alpha_i
                            max_abs_sum = q_sum.norm(float('inf')).item()
                            if max_abs_sum < 1:
                                self.n_bit += par_buf.numel() * self.n_fake_workers
                            else:
                                self.n_bit += par_buf.numel() * (
                                            1 + np.ceil(np.log2(max_abs_sum))) * self.n_fake_workers
                        else:
                            self.n_bit += par_buf.numel() * 32 * self.n_fake_workers

                    ################################
                    if not self.layerwise:
                        if self.alpha is not None:
                            self.max_sum_coord = max(self.max_sum_coord, par_buf.norm(
                                float('inf')).item() * self.n_fake_workers * self.alpha)
                        update_norm_sq += par_buf.norm().item() ** 2
                    else:
                        alpha_i = self.all_alpha[i]
                        if alpha_i is not None:
                            self.max_sum_coord = max(self.max_sum_coord,
                                                     par_buf.norm(
                                                         float('inf')).item() * self.n_fake_workers * alpha_i)
                        update_norm_sq = par_buf.norm().item() ** 2
                        self.r[i] *= self.beta
                        if self.r[i] == 0:
                            self.r[i] = update_norm_sq
                        else:
                            self.r[i] += (1 - self.beta) * update_norm_sq
                        self.all_alpha[i] = self.alpha_coef * np.sqrt(
                            dim_i / (2 * self.n_fake_workers * self.r[i] + (
                                    dim_i / self.total_dim) * self.sigma_sq) + torch.finfo(float).eps)
                    ################################
                    par_buf.zero_()
            if self.i_worker % self.n_fake_workers == self.n_fake_workers - 1:
                "The moment when all quantized updates are averaged and alpha is updated"
                ################################
                if not self.layerwise:
                    self.r *= self.beta
                    if self.r == 0:
                        self.r = (1 - self.beta) * update_norm_sq
                    else:
                        self.r += (1 - self.beta) * update_norm_sq
                    self.alpha = self.alpha_coef * np.sqrt(
                        self.total_dim / (2 * self.n_fake_workers * self.r + self.sigma_sq) + torch.finfo(float).eps)
                    self.alphas.append(self.alpha)
                ################################
                if self.max_coord > 0:
                    self.max_coords.append(self.max_coord)
                    self.max_sum_coords.append(self.max_sum_coord)

                self.n_mbs.append(self.n_bit / (8 * 1e6))
                self.max_coord = 0
                self.max_sum_coord = 0

        self.i_worker += 1

        return loss
