import torch

n_fake_workers = 8
noisy_train_stats = False

weight_decay = 0
coordwise = False
nesterov = False
momentum=0
dampening=0
seeds = [88508]
# seeds = [88508, 340925, 173557, 276948]
lr_decay_epoches = [120, 160]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_train = 50000