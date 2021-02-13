import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np


class HintSGD(Optimizer):
    r"""Implements heuristic-based IntSGD
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False,
                 n_fake_workers=8):

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
        super(HintSGD, self).__init__(params, defaults)

        self.i_worker = 0
        self.n_fake_workers = n_fake_workers
        self.max_coords = []
        self.max_coord = 0
        self.max_sum_coords = []
        self.max_sum_coord = 0
        self.n_mbs = []
        self.n_bit = 0
        self.max_int_lc = 0
        self.max_int = 0
        self.alphas = []

    def __setstate__(self, state):
        super(HintSGD, self).__setstate__(state)
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

            # profiling
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad
                lg = torch.clone(d_p).detach()
                # the largest integer on a local worker
                self.max_int_lc = max(self.max_int_lc, lg.norm(float('inf')).item())

            # aggregates the largest integers on different workers
            self.max_int += self.max_int_lc
            # self.max_int = max(self.max_int, self.max_int_lc)

            if self.i_worker % self.n_fake_workers == self.n_fake_workers - 1:
                alpha = 2 ** 10 / self.max_int
                self.max_int = 0
                self.max_int_lc = 0
            else:
                alpha = None

            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad
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

                if 'parallel_buffer' not in param_state or param_state['parallel_buffer'] is None:
                    par_buf = param_state['parallel_buffer'] = torch.clone(d_p).detach().unsqueeze(0)
                else:
                    par_buf = param_state['parallel_buffer']
                    par_buf = param_state['parallel_buffer'] = torch.cat((par_buf, torch.clone(d_p).detach().unsqueeze(0)))

                if self.i_worker % self.n_fake_workers == self.n_fake_workers - 1:
                    "The parameters are only updated when each fake worker computed and quantized its updated"
                    Q_grads = torch.round(alpha * par_buf)
                    # max_abs_i = torch.amax(torch.abs(Q_grads), dim=tuple(Q_grads.shape)[1:])
                    max_abs_i, _ = torch.abs(Q_grads).view(Q_grads.size(0), -1).max(dim=-1)
                    max_abs_i[max_abs_i < 1] = 1
                    self.n_bit += torch.clone(d_p).detach().numel() * (
                            self.n_fake_workers + torch.sum(torch.ceil(torch.log2(max_abs_i))).item())
                    self.max_coord = max_abs_i.norm(float('inf')).item()
                    Q_grad_sum = torch.sum(Q_grads, dim=0)
                    Q_grad_avg = torch.mean(Q_grads, dim=0) / alpha
                    p.add_(Q_grad_avg, alpha=-group['lr'])
                    self.max_sum_coord = max_abs_sum = Q_grad_sum.norm(float('inf')).item()
                    if max_abs_sum < 1:
                        self.n_bit += torch.clone(d_p).detach().numel() * self.n_fake_workers
                    else:
                        self.n_bit += torch.clone(d_p).detach().numel() * (
                                1 + np.ceil(np.log2(max_abs_sum))) * self.n_fake_workers
                    # par_buf.zero_()
                    param_state['parallel_buffer'] = None

            if self.i_worker % self.n_fake_workers == self.n_fake_workers - 1:
                "The moment when all quantized updates are averaged and alpha is updated"
                self.n_mbs.append(self.n_bit / (8 * 1e6))
                self.alphas.append(alpha)
                self.max_coords.append(self.max_coord)
                self.max_sum_coords.append(self.max_sum_coord)
                self.max_coord = 0
                self.max_sum_coord = 0

        self.i_worker += 1

        return loss
