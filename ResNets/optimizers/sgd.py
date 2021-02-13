import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np


class SGD(Optimizer):
    r"""Implements stochastic gradient descent
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, n_fake_workers=8):

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
        super(SGD, self).__init__(params, defaults)

        self.i_worker = 0
        self.n_fake_workers = n_fake_workers
        #####################
        # self.total_dim = total_dim
        #####################
        # self.beta = beta
        self.max_coords = []
        self.max_coord = 0
        self.max_sum_coords = []
        self.max_sum_coord = 0
        self.n_mbs = []
        self.n_bit = 0

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
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
            # update_norm_sq = 0

            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad
                # dim_i = np.prod(list(d_p.shape))
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
                    update = torch.clone(d_p).detach()
                    self.n_bit += update.numel() * 32
                    par_buf.add_(update, alpha=1 / self.n_fake_workers)

                if self.i_worker % self.n_fake_workers == self.n_fake_workers - 1:
                    "The parameters are only updated when each fake worker computed and quantized its updated"
                    p.add_(par_buf, alpha=-group['lr'])
                    self.n_bit += par_buf.numel() * 32 * self.n_fake_workers
                    par_buf.zero_()
            if self.i_worker % self.n_fake_workers == self.n_fake_workers - 1:
                "The moment when all quantized updates are averaged and alpha is updated"
                self.n_mbs.append(self.n_bit / (8 * 1e6))

        self.i_worker += 1

        return loss
