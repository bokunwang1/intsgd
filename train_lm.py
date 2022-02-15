#!/usr/bin/env python3

import datetime
import os
import re
import time
import random
from pathlib import Path
import numpy as np
import torch

import gradient_reducers
import tasks
from mean_accumulator import MeanAccumulator
from timer import Timer

config = dict(
    distributed_backend="nccl",
    num_epochs=90,
    optimizer_batch_size=64,
    optimizer_conv_learning_rate=1.25,
    optimizer_learning_rate=1.25,
    optimizer_decay_at_epochs=[100],
    optimizer_decay_with_factor=10.0,
    optimizer_memory=False,
    optimizer_momentum_type="nesterov",
    optimizer_momentum=0.0,
    # choose reducer
    optimizer_reducer="ExactReducer",
    optimizer_scale_lr_with_factor=16,  # set to override world_size as a factor
    optimizer_scale_lr_with_warmup_epochs=5,  # scale lr by world size
    optimizer_weight_decay_conv=0.0,
    optimizer_weight_decay_other=0.0,
    optimizer_weight_decay_bn=0.0,
    optimizer_mom_before_reduce=False,
    optimizer_wd_before_reduce=False,
    task="LanguageModeling",
    task_architecture="LSTM",
    seed=42,
    distributed_init_file=None,
    log_verbosity=2,
    # For PowerSGD
    optimizer_reducer_rank=4,
    optimizer_reducer_reuse_query=True,
    optimizer_reducer_n_power_iterations=0,
    # for Intsgd
    optimizer_reducer_alpha0=1.0,
    optimizer_reducer_alpha=100.0,
    optimizer_reducer_rand_round=False,
    optimizer_reducer_int=True,
    optimizer_reducer_beta=0.9,
    # for multi-node multi-proc
    rank=0,
    n_workers=1,
    local_rank=0,
    local_world_size=1,
)


def main():
    output_dir = "../output"
    seed = int(config["seed"])
    rank = int(config["rank"])
    n_workers = int(config["n_workers"])
    seed_everything(seed + rank)
    print('rank:{0}/{1}, local rank:{2}/{3}'.format(config["rank"], config["n_workers"], config["local_rank"],
                                                    config["local_world_size"]))

    print('rank: {0}, available devices:{1}'.format(config["rank"], torch.cuda.device_count()))

    device = torch.device("cuda:" + str(config["local_rank"]) if torch.cuda.is_available() else "cpu")
    print('rank: {0}, current device:{1}'.format(config["rank"], device))
    timer = Timer(verbosity_level=config["log_verbosity"], log_fn=metric)

    if torch.distributed.is_available():
        if config["distributed_init_file"] is None:
            config["distributed_init_file"] = os.path.join(output_dir, "dist_init")
        print(
            "Distributed init: rank {}/{} - {}".format(
                config["rank"], config["n_workers"], config["distributed_init_file"]
            )
        )
        torch.distributed.init_process_group(
            backend=config["distributed_backend"],
            init_method="file://" + os.path.abspath(config["distributed_init_file"]),
            timeout=datetime.timedelta(seconds=120),
            world_size=n_workers,
            rank=rank,
        )
    task = tasks.build(task_name=config["task"], device=device, timer=timer, **config)
    # calculate total dim here
    total_dim = get_total_dim(task.state)
    n_layers = len(task.state)
    reducer = get_reducer(device, timer, total_dim, n_layers)

    bits_communicated = 0
    memories = [torch.zeros_like(param) for param in task.state]
    momenta = [torch.empty_like(param) for param in task.state]
    send_buffers = [torch.zeros_like(param) for param in task.state]

    # collect info
    all_test_losses = []
    all_test_perps = []
    all_alphas = []
    all_bytes_communicated = []

    for epoch in range(config["num_epochs"]):
        print("state.progress: {0}/{1}, current epoch:{2}".format(float(epoch), config["num_epochs"], epoch))

        task.update_epoch(epoch)

        task.init_hidden()

        # Determine per-parameter optimization parameters
        wds = [get_weight_decay(epoch, name) for name in task.parameter_names]

        iters_per_batch = range(0, task.train_data.size(0) - 1, task.bptt)

        for batch, i in enumerate(iters_per_batch):
            data, target = task.train_iterator(i)
            epoch_frac = epoch + i / len(iters_per_batch)
            lrs = [get_learning_rate(epoch_frac, name) for name in task.parameter_names]

            with timer("batch", epoch_frac):

                _, grads, _ = task.batch_loss_and_gradient(data, target)

                if config["optimizer_wd_before_reduce"]:
                    with timer("batch.weight_decay", epoch_frac, verbosity=2):
                        for grad, param, wd in zip(grads, task.state, wds):
                            if wd > 0:
                                grad.add_(param.detach(), alpha=wd)

                if config["optimizer_mom_before_reduce"]:
                    with timer("batch.momentum", epoch_frac, verbosity=2):
                        for grad, momentum in zip(grads, momenta):
                            if epoch == 0 and i == 0:
                                momentum.data = grad.clone().detach()
                            else:
                                if (
                                        config["optimizer_momentum_type"]
                                        == "exponential_moving_average"
                                ):
                                    momentum.mul_(config["optimizer_momentum"]).add_(
                                        grad, alpha=1 - config["optimizer_momentum"]
                                    )
                                else:
                                    momentum.mul_(config["optimizer_momentum"]).add_(grad)
                            replace_grad_by_momentum(grad, momentum)

                with timer("batch.accumulate", epoch_frac, verbosity=2):
                    for grad, memory, send_bfr in zip(grads, memories, send_buffers):
                        if config["optimizer_memory"]:
                            send_bfr.data[:] = grad + memory
                        else:
                            send_bfr.data[:] = grad

                with timer("batch.reduce", epoch_frac):
                    bits_communicated += reducer.reduce(send_buffers, grads, memories)

                if not config["optimizer_wd_before_reduce"]:
                    with timer("batch.wd", epoch_frac, verbosity=2):
                        for grad, param, wd in zip(grads, task.state, wds):
                            if wd > 0:
                                grad.add_(param.detach(), alpha=wd)

                if not config["optimizer_mom_before_reduce"]:
                    with timer("batch.mom", epoch_frac, verbosity=2):
                        for grad, momentum in zip(grads, momenta):
                            if epoch == 0 and i == 0:
                                momentum.data = grad.clone().detach()
                            else:
                                if (
                                        config["optimizer_momentum_type"]
                                        == "exponential_moving_average"
                                ):
                                    momentum.mul_(config["optimizer_momentum"]).add_(
                                        grad, alpha=1 - config["optimizer_momentum"]
                                    )
                                else:
                                    momentum.mul_(config["optimizer_momentum"]).add_(grad)
                            replace_grad_by_momentum(grad, momentum)

                with timer("batch.step", epoch_frac, verbosity=2):
                    for param, grad, lr in zip(task.state, grads, lrs):
                        param.data.add_(grad, alpha=-lr)

        with timer("test.last", epoch):
            test_stats = task.test()

            all_test_info = test_stats
            if config["optimizer_reducer"] in ["IntQuantReducer"]:
                if torch.is_tensor(reducer.alpha):
                    alpha_val = reducer.alpha.item()
                else:
                    alpha_val = reducer.alpha
                all_alphas.append(alpha_val)

            if torch.is_tensor(all_test_info['cross_entropy']):
                ce_val = all_test_info['cross_entropy'].item()
            else:
                ce_val = all_test_info['cross_entropy']

            if torch.is_tensor(all_test_info['perplexity']):
                perp_val = all_test_info['perplexity'].item()
            else:
                perp_val = all_test_info['perplexity']
            all_test_losses.append(ce_val)
            all_test_perps.append(perp_val)
            all_bytes_communicated.append(bits_communicated / (8 * 1e6))

        if torch.distributed.get_rank() == 0:
            print("Epoch: {0}, Test loss: {1}, test perp: {2}".format(epoch, ce_val, perp_val))
            method_name = config['optimizer_reducer']
            if config["optimizer_reducer"] == "RankKReducer":
                method_name += ('_' + str(config['optimizer_memory']))
            elif config["optimizer_reducer"] == "IntQuantReducer":
                method_name += ('_' + str(config['optimizer_reducer_rand_round']))
                method_name += ('_' + str(config['optimizer_overflow_handling']))
                method_name += ('_' + str(config['optimizer_reducer_int']))
            elif config["optimizer_reducer"] == "HintQuantReducer":
                method_name += ('_' + str(config['optimizer_reducer_rand_round']))
                method_name += ('_' + str(config['optimizer_overflow_handling']))
                method_name += ('_' + str(config['optimizer_reducer_int']))
            fl_name = config['task_architecture'] + "_" + method_name + "_" + str(seed) + "_" + str(
                config["n_workers"]) + "_timer_summary.json"
            timer.save_summary(os.path.join(output_dir, fl_name))

    method_name = config['optimizer_reducer']
    if config["optimizer_reducer"] == "RankKReducer":
        method_name += ('_' + str(config['optimizer_memory']))
    elif config["optimizer_reducer"] == "IntQuantReducer":
        method_name += ('_' + str(config['optimizer_reducer_rand_round']))
        method_name += ('_' + str(config['optimizer_overflow_handling']))
        method_name += ('_' + str(config['optimizer_reducer_int']))
    elif config["optimizer_reducer"] == "HintQuantReducer":
        method_name += ('_' + str(config['optimizer_reducer_rand_round']))
        method_name += ('_' + str(config['optimizer_overflow_handling']))
        method_name += ('_' + str(config['optimizer_reducer_int']))
    save_results(mbs=np.array(all_bytes_communicated), test_losses=np.array(all_test_losses),
                 test_perp=np.array(all_test_perps), seed=seed, n_workers=config['n_workers'],
                 all_alphas=np.array(all_alphas), method_name=method_name, experiment=config['task_architecture'])


def save_results(mbs, test_losses, test_perp, seed, n_workers, all_alphas=None, method_name='ExactReducer',
                 experiment='ResNet18',
                 folder='output'):
    if all_alphas is None:
        all_alphas = []
    path_folder = './{0}/{1}-{2}/{3}'.format(folder, experiment, n_workers, seed)
    Path(path_folder).mkdir(parents=True, exist_ok=True)
    path = path_folder + '_' + method_name + '_'
    to_save = [mbs, test_losses, test_perp, all_alphas]
    prefixes = ['mbs', 'te_l', 'te_acc', 'al']  # "te_acc" is only for the convenience of plotting ...
    for log, prefix in zip(to_save, prefixes):
        np.save(path + prefix + '.npy', log)


def get_weight_decay(epoch, parameter_name):
    """Take care of differences between weight decay for parameters"""
    if is_conv_param(parameter_name):
        return config["optimizer_weight_decay_conv"]
    elif is_batchnorm_param(parameter_name):
        return config["optimizer_weight_decay_bn"]
    else:
        return config["optimizer_weight_decay_other"]


def get_learning_rate(epoch, parameter_name):
    """Apply any learning rate schedule"""
    if is_conv_param(parameter_name):
        lr = config["optimizer_conv_learning_rate"]
    else:
        lr = config["optimizer_learning_rate"]

    if config["optimizer_scale_lr_with_warmup_epochs"]:
        warmup_epochs = config["optimizer_scale_lr_with_warmup_epochs"]
        max_factor = config.get("optimizer_scale_lr_with_factor", None)
        if max_factor is None:
            max_factor = (
                torch.distributed.get_world_size() if torch.distributed.is_available() else 1.0
            )
        factor = 1.0 + (max_factor - 1.0) * min(epoch / warmup_epochs, 1.0)
        lr *= factor

    for decay_epoch in config["optimizer_decay_at_epochs"]:
        if epoch >= decay_epoch:
            lr /= config["optimizer_decay_with_factor"]
        else:
            return lr
    return lr


def is_conv_param(parameter_name):
    """
    Says whether this parameter is a conv linear layer that
    needs a different treatment from the other weights
    """
    return "conv" in parameter_name and "weight" in parameter_name


def is_batchnorm_param(parameter_name):
    """
    Is this parameter part of a batchnorm parameter?
    """
    return re.match(r""".*\.bn\d+\.(weight|bias)""", parameter_name)


def replace_grad_by_momentum(grad, momentum):
    """
    Inplace operation that applies momentum to a gradient.
    This distinguishes between types of momentum (heavy-ball vs nesterov)
    """
    if config["optimizer_momentum_type"] == "heavy-ball":
        grad.data[:] = momentum
    if config["optimizer_momentum_type"] == "exponential_moving_average":
        grad.data[:] = momentum
    elif config["optimizer_momentum_type"] == "nesterov":
        grad.data[:] += momentum
    else:
        raise ValueError("Unknown momentum type")


def get_reducer(device, timer, total_dim, n_layers):
    """Configure the reducer from the config"""
    if config["optimizer_reducer"] == "RankKReducer":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
            n_power_iterations=config["optimizer_reducer_n_power_iterations"],
            reuse_query=config["optimizer_reducer_reuse_query"],
            rank=config["optimizer_reducer_rank"],
        )
    elif config["optimizer_reducer"] == "IntQuantReducer":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
            alpha=config["optimizer_reducer_alpha"],
            beta=config["optimizer_reducer_beta"],
            alpha0=config["optimizer_reducer_alpha0"],
            rand_round=config["optimizer_reducer_rand_round"],
            overflow_handling=config["optimizer_overflow_handling"],
            int8=config["optimizer_reducer_int"],
            total_dim=total_dim,
        )
    elif config["optimizer_reducer"] == "HintQuantReducer":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
            rand_round=config["optimizer_reducer_rand_round"],
            overflow_handling=config["optimizer_overflow_handling"],
            int8=config["optimizer_reducer_int"],
        )
    elif config["optimizer_reducer"] == "Uniform_Quant":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
        )
    elif config["optimizer_reducer"] == "Natural_Quant":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
            local_rank=config["local_rank"],
        )
    else:
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            config["seed"], device, timer
        )

def seed_everything(seed=1029):
    '''
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


@torch.jit.script
def l2norm(tensor):
    """Compute the L2 Norm of a tensor in a fast and correct way"""
    # tensor.norm(p=2) is buggy in Torch 1.0.0
    # tensor.norm(p=2) is really slow in Torch 1.0.1
    return torch.sqrt(torch.sum(tensor ** 2))


def log_metric(name, values, tags={}):
    """Log timeseries data
       This function will be overwritten when called through run.py"""
    value_list = []
    for key in sorted(values.keys()):
        value = values[key]
        value_list.append(f"{key}:{value:7.3f}")
    values = ", ".join(value_list)
    tag_list = []
    for key, tag in tags.items():
        tag_list.append(f"{key}:{tag}")
    tags = ", ".join(tag_list)
    print("{name:30s} - {values} ({tags})".format(name=name, values=values, tags=tags))


def metric(*args, **kwargs):
    if int(config["rank"]) == 0:
        log_metric(*args, **kwargs)


def get_total_dim(tsr_lst):  # input: a list of tensors
    dim = 0
    for tsr in tsr_lst:
        nn = 1
        for s in list(tsr.size()):
            nn = nn * s
        dim += nn
    return dim


if __name__ == "__main__":
    main()
