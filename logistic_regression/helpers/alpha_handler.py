import numpy as np

class Alpha_Handler(object):
    def __init__(self, alpha_choice, lr, sigma_Q, num_workers, beta, alpha0):
        super().__init__()
        self.alpha_choice = alpha_choice
        self.lr = lr
        self.sigma_Q = sigma_Q
        self.num_workers = num_workers
        self.beta = beta
        self.alpha0 = alpha0
        self.alpha = alpha0
        self.r_k = 0

    def get_alpha(self):
        return self.alpha

    def reset_alpha(self):
        self.alpha = self.alpha0
        self.r_k = 0

    def set_init_r_k(self, x, x_prev):
        self.r_k = np.linalg.norm(x - x_prev)**2

    def set_alpha(self, x, x_prev):
        lr = self.lr.get_lr()
        if self.alpha_choice == 'constant':
            self.alpha = (np.sqrt(len(x))/self.sigma_Q) * np.ones_like(x)
        elif self.alpha_choice == 'adaptive':
            alpha = lr*np.sqrt(len(x))/((np.sqrt(2*self.num_workers)*np.linalg.norm(x - x_prev)) + np.finfo(float).eps)
            self.alpha = alpha * np.ones_like(x)
        elif self.alpha_choice == 'moving_avg':
            r_k_prev= self.r_k.copy()
            self.r_k = self.beta*r_k_prev + (1-self.beta)*np.linalg.norm(x - x_prev)**2
            alpha = lr*np.sqrt(len(x)) / (np.sqrt(2*self.num_workers*self.r_k) + np.finfo(float).eps)
            self.alpha = alpha * np.ones_like(x)
        elif self.alpha_choice == 'moving_avg_w_const':
            r_k_prev= self.r_k.copy()
            self.r_k = self.beta*r_k_prev + (1-self.beta)*np.linalg.norm(x - x_prev)**2
            alpha = lr*np.sqrt(len(x)) /np.sqrt(2*self.num_workers*self.r_k + (lr**2)*(self.sigma_Q**2))
            self.alpha = alpha * np.ones_like(x)

