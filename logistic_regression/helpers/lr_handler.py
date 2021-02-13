import numpy as np


class LR_Handler(object):
    def __init__(self, num_iters, decay, lr0=1):
        super().__init__()
        self.num_iters = num_iters
        self.decay = decay
        self.lr0 = lr0
        self.lr = lr0

    def get_lr(self):
        return self.lr

    def set_lr(self):
        if not self.decay:
            self.lr = self.lr0
        else:
            self.lr = self.lr0 / np.sqrt(self.num_iters)
