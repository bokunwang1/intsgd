from optimizers.sgd import SGD
from optimizers.intsgd import IntSGD
from optimizers.natsgd import NatSGD
from optimizers.intsgd_h import HintSGD
from models import densenet, lenet, resnet
from utils import load_data, save_results, seed_everything, get_n_params
import argparse
from train import run_sgd
from config import weight_decay, coordwise, nesterov, momentum, dampening, seeds, lr_decay_epoches, \
    device, N_train, n_fake_workers, noisy_train_stats
import numpy as np

parser = argparse.ArgumentParser(description='Run the algorithms')
parser.add_argument('--n_epoch', action='store', dest='n_epoch', type=int, help='Number of epoches')
parser.add_argument('--beta', action='store', dest='beta', default=0.0, type=float, help='Value of beta')
parser.add_argument('--sigma_sq', action='store', dest='sigma_sq', default=0.0, type=float, help='Value of sigma_sq')
parser.add_argument('--alg', action='store', dest='algo_name', type=str,
                    help='Which algorithm: SGD or IntSGD')
parser.add_argument('--lr0', action='store', dest='lr0', type=float, help='Value of initial stepsize')
parser.add_argument('--bs', action='store', dest='batch_size', type=int, help='Batch size for stochastic algorithms')
parser.add_argument('--net', action='store', dest='experiment', type=str, help='lenet, resnet or resnet50')
parser.add_argument('--q', action='store', dest='q_type', type=str, default='single', help='layerwise or single')
parser.add_argument('--r', action='store', dest='r_type', type=str, default='det', help='rand or det')
parser.add_argument('--alpha0', action='store', dest='alpha_coef', default=1.0, type=float, help='alpha coefficient')

args = parser.parse_args()
n_epoch = args.n_epoch
beta = args.beta
sigma_sq = args.sigma_sq
algo_name = args.algo_name
lr0 = args.lr0
batch_size = args.batch_size
experiment = args.experiment
q_type = args.q_type
r_type = args.r_type
alpha_coef = args.alpha_coef

if q_type == 'layerwise':
    layerwise = True
else:
    layerwise = False

if r_type == 'rand':
    random_round = True
else:
    random_round = False

if experiment == 'densenet':
    net_class = densenet.DenseNet121
elif experiment == 'resnet':
    net_class = resnet.ResNet18
elif experiment == 'lenet':
    net_class = lenet.LeNet
elif experiment == 'resnet50':
    net_class = resnet.ResNet50
else:
    raise ValueError('Selected network is not supported!')

if algo_name == 'SGD':
    trainloader, testloader, num_classes = load_data(batch_size=batch_size)
    for r, seed in enumerate(seeds):
        seed_everything(seed)
        epoch_lrs = [(lr_decay_epoches[0], lr0 / 10), (lr_decay_epoches[1], lr0 / 100)]
        net = net_class()
        net.to(device)
        opt = SGD(net.parameters(), lr=lr0, momentum=momentum, n_fake_workers=n_fake_workers)
        losses_sgd, test_loss_sgd, train_acc_sgd, test_acc_sgd, it_train_sgd, it_test_sgd, grad_norms_sgd = run_sgd(
            net=net, device=device, trainloader=trainloader, testloader=testloader, N_train=N_train,
            batch_size=batch_size, n_epoch=n_epoch, optimizer=opt, epoch_lrs=epoch_lrs,
            noisy_train_stat=noisy_train_stats,
            checkpoint=N_train // batch_size // 3
        )
        method = '{0}_{1}_sgd_lr_{2}'.format(experiment, str(batch_size), lr0)
        for (epoch, lr) in epoch_lrs:
            method += '_{}_{}'.format(epoch, lr)
        method = (method + '_run_{}'.format(r))
        n_mbs = opt.n_mbs
        idx = np.arange(0, len(n_mbs), (len(n_mbs)//len(it_train_sgd)))
        # n_mbs = list(np.array(n_mbs)[idx])[1:]
        n_mbs = list(np.array(n_mbs)[idx])
        save_results(losses_sgd, test_loss_sgd, train_acc_sgd, test_acc_sgd, it_train_sgd, it_test_sgd, grad_norms_sgd,
                     method=method, experiment=experiment, n_mbs=n_mbs)
elif algo_name == 'NatSGD':
    trainloader, testloader, num_classes = load_data(batch_size=batch_size)
    for r, seed in enumerate(seeds):
        seed_everything(seed)
        epoch_lrs = [(lr_decay_epoches[0], lr0 / 10), (lr_decay_epoches[1], lr0 / 100)]
        net = net_class()
        net.to(device)
        opt = NatSGD(net.parameters(), lr=lr0, momentum=momentum, n_fake_workers=n_fake_workers)
        losses_nat, test_loss_nat, train_acc_nat, test_acc_nat, it_train_nat, it_test_nat, grad_norms_nat = run_sgd(
            net=net, device=device, trainloader=trainloader, testloader=testloader, N_train=N_train,
            batch_size=batch_size, n_epoch=n_epoch, optimizer=opt, epoch_lrs=epoch_lrs,
            noisy_train_stat=noisy_train_stats,
            checkpoint=N_train // batch_size // 3
        )
        method = '{0}_{1}_nat_sgd_lr_{2}'.format(experiment, str(batch_size), lr0)
        for (epoch, lr) in epoch_lrs:
            method += '_{}_{}'.format(epoch, lr)
        method = (method + '_run_{}'.format(r))
        n_mbs = opt.n_mbs
        idx = np.arange(0, len(n_mbs), (len(n_mbs) // len(it_train_nat)))
        # n_mbs = list(np.array(n_mbs)[idx])[1:-1]
        n_mbs = list(np.array(n_mbs)[idx])
        save_results(losses_nat, test_loss_nat, train_acc_nat, test_acc_nat, it_train_nat, it_test_nat, grad_norms_nat,
                     method=method, experiment=experiment, n_mbs=n_mbs)
elif algo_name == 'HintSGD':
    trainloader, testloader, num_classes = load_data(batch_size=batch_size)
    for r, seed in enumerate(seeds):
        seed_everything(seed)
        epoch_lrs = [(lr_decay_epoches[0], lr0 / 10), (lr_decay_epoches[1], lr0 / 100)]
        net = net_class()
        net.to(device)
        opt = HintSGD(net.parameters(), lr=lr0, momentum=momentum, n_fake_workers=n_fake_workers)
        losses_hint, test_loss_hint, train_acc_hint, test_acc_hint, it_train_hint, it_test_hint, grad_norms_hint = run_sgd(
            net=net, device=device, trainloader=trainloader, testloader=testloader, N_train=N_train,
            batch_size=batch_size, n_epoch=n_epoch, optimizer=opt, epoch_lrs=epoch_lrs,
            noisy_train_stat=noisy_train_stats,
            checkpoint=N_train // batch_size // 3
        )
        method = '{0}_{1}_hint_sgd_lr_{2}'.format(experiment, str(batch_size), lr0)
        for (epoch, lr) in epoch_lrs:
            method += '_{}_{}'.format(epoch, lr)
        method = (method + '_run_{}'.format(r))
        n_mbs = opt.n_mbs
        alphas = opt.alphas
        idx = np.arange(0, len(n_mbs), (len(n_mbs) // len(it_train_hint)))
        # n_mbs = list(np.array(n_mbs)[idx])[1:-1]
        n_mbs = list(np.array(n_mbs)[idx])
        save_results(losses_hint, test_loss_hint, train_acc_hint, test_acc_hint, it_train_hint, it_test_hint, grad_norms_hint,
                     method=method, experiment=experiment, n_mbs=n_mbs, alphas=alphas)
elif algo_name == 'IntSGD':
    trainloader, testloader, num_classes = load_data(batch_size=batch_size)
    for r, seed in enumerate(seeds):
        seed_everything(seed)
        epoch_lrs = [(lr_decay_epoches[0], lr0 / 10), (lr_decay_epoches[1], lr0 / 100)]
        net = net_class()
        net.to(device)
        dimension = get_n_params(net)
        opt = IntSGD(net.parameters(), total_dim=dimension, lr=lr0, momentum=momentum, dampening=dampening,
                     weight_decay=weight_decay, nesterov=nesterov, n_fake_workers=n_fake_workers, layerwise=layerwise,
                     coordwise=coordwise, sigma_sq=sigma_sq, random_round=random_round, alpha_coef=alpha_coef,
                     beta=beta)
        losses_int, test_loss_int, train_acc_int, test_acc_int, it_train_int, it_test_int, grad_norms_int = run_sgd(
            net=net, device=device, trainloader=trainloader, testloader=testloader, N_train=N_train,
            batch_size=batch_size, n_epoch=n_epoch, optimizer=opt, epoch_lrs=epoch_lrs,
            noisy_train_stat=noisy_train_stats,
            checkpoint=N_train // batch_size // 3
        )
        max_coords = opt.max_coords
        max_sum_coords = opt.max_sum_coords
        if layerwise == True:
            alphas = []
        else:
            alphas = opt.alphas
        n_mbs = opt.n_mbs
        idx = np.arange(0, len(n_mbs), (len(n_mbs) // len(it_train_int)))
        # n_mbs = list(np.array(n_mbs)[idx])[1:-1]
        n_mbs = list(np.array(n_mbs)[idx])
        method = '{0}_{1}_int_layerwise_{2}_rand_round_{3}_beta_{4}_sigma_sq_{5}_lr_{6}'.format(experiment, alpha_coef,
                                                                                                layerwise, random_round,
                                                                                                beta, sigma_sq, lr0)
        for (epoch, lr) in epoch_lrs:
            method += '_{}_{}'.format(epoch, lr)
        method = (method + '_run_{}'.format(r))
        save_results(losses_int, test_loss_int, train_acc_int, test_acc_int, it_train_int, it_test_int, grad_norms_int,
                     max_coords=max_coords, max_sum_coords=max_sum_coords, method=method, experiment=experiment,
                     n_mbs=n_mbs, alphas=alphas)
