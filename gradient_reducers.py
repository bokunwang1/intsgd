import datetime
import os
import time
from contextlib import contextmanager
from typing import List

import numpy as np
import torch

class Reducer:
    def __init__(self, random_seed, device, timer):
        self.rng = np.random.RandomState(random_seed)
        M = 1024 * 1024
        self.precalc_numbers = (
            torch.from_numpy(self.rng.randn(128 * M)).to(device).type(torch.float32)
        )
        if torch.distributed.is_available():
            self.n_workers = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.n_workers = 1
            self.rank = 0
        self.device = device
        self.timer = timer

    def reduce(self, grad_in, grad_out, memory_out):
        """Return communicated bits"""
        raise NotImplementedError()


class RankKReducer(Reducer):
    def __init__(self, random_seed, device, timer, n_power_iterations=0, reuse_query=False, rank=1):
        super().__init__(random_seed, device, timer)
        assert n_power_iterations == 0
        self.rank = rank
        self.p_memory = None
        self.q_memory = None
        self.reuse_query = reuse_query

    def set_random(self, vector):
        torch.manual_seed(self.rng.randint(1_000_000_000))
        vector.data[:] = torch.randn(*vector.shape, device=self.device)
        # orthogonalize(vector)

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0

        # Split the tensors into rank1-ones that will be reduced un-compressed
        # and rank > 1 tensors that are compressed
        rank1_tensors = [
            (tensor, out, mem)
            for tensor, out, mem in zip(grad_in, grad_out, memory_out)
            if tensor.ndimension() <= 1
        ]
        high_rank_tensors = [
            (tensor, out, mem)
            for tensor, out, mem in zip(grad_in, grad_out, memory_out)
            if tensor.ndimension() > 1
        ]

        # We are building a rank-1 approximation of every tensor
        # that can be interpreted as a matrix. Let the approximation be
        # M = p q^T
        # We are allocating consequtive memory for the p's and q's

        memory_is_uninitialized = self.p_memory is None

        with self.timer("reduce.allocate_memory", verbosity=2):
            p_total_size = 0
            q_total_size = 0
            for tensor, _, _ in high_rank_tensors:
                matrix = tensor.view(tensor.shape[0], -1)
                n, m = matrix.shape
                rank = min(n, m, self.rank)
                p_total_size += n * rank
                q_total_size += m * rank
            if self.p_memory is None:
                self.p_memory = torch.empty(p_total_size, device=self.device)
                self.q_memory = torch.empty(q_total_size, device=self.device)

            # Find them again and make lists of pointers
            ps = []
            qs = []
            p_idx = 0
            q_idx = 0
            for tensor, _, _ in high_rank_tensors:
                matrix = tensor.view(tensor.shape[0], -1)
                n, m = matrix.shape
                rank = min(n, m, self.rank)
                ps.append(self.p_memory[p_idx: p_idx + n * rank].view(n, rank))
                qs.append(self.q_memory[q_idx: q_idx + m * rank].view(m, rank))
                p_idx += n * rank
                q_idx += m * rank

        with self.timer("reduce.prepare.q", verbosity=2):
            for (tensor, _, _), q, p in zip(high_rank_tensors, qs, ps):
                matrix = tensor.view(tensor.shape[0], -1)
                n, m = matrix.shape

                if self.reuse_query and not memory_is_uninitialized:
                    # orthogonalize(q)
                    pass
                else:
                    # Sample a query vector q
                    self.set_random(q)

        with self.timer("reduce.compute.p", verbosity=2):
            for (tensor, _, _), q, p in zip(high_rank_tensors, qs, ps):
                matrix = tensor.view(tensor.shape[0], -1)
                torch.matmul(matrix, q, out=p)

        with self.timer("reduce.p", verbosity=2):
            all_reduce(self.p_memory)
            bits_communicated += n_bits(self.p_memory)
            self.p_memory.data[:] /= self.n_workers

        # Start communicating rank 1 tensors
        with self.timer("reduce.rank1.pack", verbosity=2):
            rank1_tensor_list = TensorBuffer([tensor for (tensor, _, _) in rank1_tensors])
        with self.timer("reduce.rank1.all_reduce", verbosity=2):
            rank1_handle = rank1_tensor_list.all_reduce(async_op=True)
            bits_communicated += rank1_tensor_list.bits()

        with self.timer("reduce.normalize.p", verbosity=2):
            for p in ps:
                orthogonalize(p)

        with self.timer("reduce.compute.q", verbosity=2):
            for p, q, (tensor, _, _) in zip(ps, qs, high_rank_tensors):
                matrix = tensor.view(tensor.shape[0], -1)
                torch.matmul(matrix.t(), p, out=q)

        with self.timer("reduce.q", verbosity=2):
            all_reduce(self.q_memory)
            bits_communicated += n_bits(self.q_memory)
            self.q_memory.data[:] /= self.n_workers

        with self.timer("reduce.outerprod", verbosity=2):
            for p, q, (tensor, out, mem) in zip(ps, qs, high_rank_tensors):
                # Set the output gradient
                torch.matmul(p, q.t(), out=out.data[:])
                mem.data[:] = tensor - out

        with self.timer("reduce.rank1.unpack", verbosity=2):
            rank1_handle.wait()
            rank1_tensor_list.buffer /= self.n_workers
            rank1_tensor_list.unpack([out for (_, out, _) in rank1_tensors])

        return bits_communicated


class ExactReducer(Reducer):
    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        with self.timer("reduce.zero_mem", verbosity=2):
            for mem in memory_out:
                mem.zero_()

        with self.timer("reduce.build_lists", verbosity=2):
            list_in = grad_in
            list_out = grad_out

        with self.timer("reduce.reduce", verbosity=2):
            bits_communicated = reduce_mean_list(self.device, list_in, list_out, self.timer)

        return bits_communicated


class Natural_Quant(Reducer):
    def __init__(self, random_seed, device, timer, local_rank):
        super().__init__(random_seed, device, timer)
        self.local_rank = local_rank

    def compress(self, tensor):
        tensor_flatten = tensor.flatten()
        tensor_cast = tensor_flatten.view(torch.int32)
        sign = tensor_cast & -2147483648
        exp = tensor_cast & 2139095040
        mantissa = tensor_cast & 8388607
        exp_add_one = mantissa > torch.randint(low=0, high=0b00000000011111111111111111111111,
                                                     size=tensor_flatten.shape,
                                                     dtype=torch.int32, device=self.device)
        exponent = torch.where(exp_add_one, exp + 0b00000000100000000000000000000000, exp)
        exp_shift = torch.clip(exponent, min=0b00001001000000000000000000000000, max=0b01001000100000000000000000000000)
        exps = exp_shift >> 23
        exps = torch.bitwise_or(sign >> 24, exps - 18)
        tensor_compressed = exps.to(torch.uint8)
        return tensor_compressed

    def decompress(self, tensor_compressed):
        sign = tensor_compressed > 127
        exps = torch.bitwise_and(tensor_compressed, 0b01111111)
        floats = ((exps + 18).to(torch.int32) << 23).view(torch.float32)
        tensor_decompressed = torch.where(sign, -floats, floats)
        tensor_decompressed = torch.multiply((exps >= 1).to(torch.float32), tensor_decompressed)
        return tensor_decompressed

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0

        with self.timer("reduce.zero_mem", verbosity=2):
            for mem in memory_out:
                mem.zero_()

        with self.timer("reduce.build_lists", verbosity=2):
            list_in = grad_in
            list_out = grad_out

        with self.timer("reduce.gather", verbosity=2):
            if torch.distributed.is_available():
                n_workers = torch.distributed.get_world_size()
            else:
                n_workers = 1

            if n_workers == 1:
                for t_in, t_out in zip(list_in, list_out):
                    t_out[:] = t_in
                return 0

            with self.timer("reduce.gather.pack"):
                values_bfr = TensorBuffer(list_in)

            with self.timer("reduce.gather.compress"):
                compressed_buffer = self.compress(values_bfr.buffer)
                values_bfr.buffer = compressed_buffer.to(torch.uint8)

            with self.timer("reduce.gather.allgather"):
                all_values_buffers = values_bfr.all_gather()
                bits_communicated += values_bfr.bits()

            with self.timer("reduce.gather.average"):
                for out in grad_out:
                    out.data[:] = 0.0

                # aggregate the results from all workers
                for values_buffer in all_values_buffers:
                    # initialize lists for unpacking
                    local_list_values = []
                    for t_in in list_in:
                        local_list_values.append(torch.empty_like(t_in))
                    decompressed = self.decompress(tensor_compressed=values_buffer)
                    values_bfr.buffer = decompressed
                    values_bfr.unpack(local_list_values)
                    for values, out in zip(local_list_values, grad_out):
                        out.add_(values.view(*out.shape), alpha=1 / self.n_workers)

        return bits_communicated


class Uniform_Quant(Reducer):
    def __init__(self, random_seed, device, timer, quantum_num=64):
        super().__init__(random_seed, device, timer)
        self.quantum_num = quantum_num

    def compress(self, tensor):
        """
        Borrowed from https://github.com/sands-lab/grace/blob/1a11c1acee8a27e19cf26fbe29b61216e3a8afba/grace_dl/torch/compressor/qsgd.py
        """
        shape = torch.tensor(tensor.size()).to(self.device).type(torch.int32)
        tensor = tensor.flatten()

        norm = tensor.norm()
        norm = norm.flatten()
        abs_gradient = tensor.abs()

        level_float = self.quantum_num / norm * abs_gradient
        previous_level = level_float.floor()
        prob = torch.empty_like(tensor).uniform_()
        is_next_level_bool = (prob < (level_float - previous_level))
        is_next_level = torch.tensor(is_next_level_bool).to(self.device).type(torch.float32)
        new_level = (previous_level + is_next_level)

        sign = tensor.sign()
        tensor_compressed = (new_level * sign).type(torch.int8)
        tensor_compressed = tensor_compressed.type(torch.int8 if self.quantum_num < 128 else torch.half)
        # tensor_compressed = tensor_compressed, norm
        return tensor_compressed, norm, shape

    def decompress(self, tensor_compressed, norm, shape):
        # tensor_compressed, norm = tensor_compressed
        shape = torch.Size(shape)
        decode_output = tensor_compressed.type(torch.float32)
        tensor_decompressed = norm / self.quantum_num * decode_output
        tensor_decompressed = tensor_decompressed.view(shape)
        return tensor_decompressed

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0

        with self.timer("reduce.zero_mem", verbosity=2):
            for mem in memory_out:
                mem.zero_()

        with self.timer("reduce.build_lists", verbosity=2):
            list_in = grad_in
            list_out = grad_out

        with self.timer("reduce.compress", verbosity=2):
            # each list item corresponds to a layer (weight) tensor
            list_values = []
            list_norm = []
            list_shape = []
            for tensor in list_in:
                values, norm, shape = self.compress(tensor)
                list_values.append(values)
                list_norm.append(norm)
                list_shape.append(shape)

        with self.timer("reduce.gather", verbosity=2):
            if torch.distributed.is_available():
                n_workers = torch.distributed.get_world_size()
            else:
                n_workers = 1

            if n_workers == 1:
                for t_compressed, norm, shape, t_out in zip(list_values, list_norm, list_shape, list_out):
                    t_out[:] = self.decompress(tensor_compressed=t_compressed, norm=norm, shape=shape)
                return 0

            with self.timer("reduce.gather.pack"):
                values_bfr = TensorBuffer(list_values)
                norm_bfr = TensorBuffer(list_norm)
                shape_bfr = TensorBuffer(list_shape)

            with self.timer("reduce.gather.allgather"):
                all_values_buffers = values_bfr.all_gather()
                bits_communicated += values_bfr.bits()
                all_norm_buffers = norm_bfr.all_gather()
                bits_communicated += norm_bfr.bits()
                all_shape_buffers = shape_bfr.all_gather()
                bits_communicated += shape_bfr.bits()

            with self.timer("reduce.gather.average"):
                for out in grad_out:
                    out.data[:] = 0.0

                # aggregate the results from all workers
                for values_buffer, norm_buffer, shape_buffer in zip(all_values_buffers, all_norm_buffers,
                                                                    all_shape_buffers):

                    # initialize lists for unpacking
                    local_list_values = []
                    local_list_norm = []
                    local_list_shape = []
                    for values, norm, shape in zip(list_values, list_norm, list_shape):
                        local_list_values.append(torch.empty_like(values))
                        local_list_norm.append(torch.empty_like(norm))
                        local_list_shape.append(torch.empty_like(shape))

                    values_bfr.buffer = values_buffer
                    values_bfr.unpack(local_list_values)
                    norm_bfr.buffer = norm_buffer
                    norm_bfr.unpack(local_list_norm)
                    shape_bfr.buffer = shape_buffer
                    shape_bfr.unpack(local_list_shape)
                    for values, norm, shape, out in zip(local_list_values, local_list_norm, local_list_shape, grad_out):
                        tensor = self.decompress(tensor_compressed=values, norm=norm, shape=shape)
                        out.add_(tensor.view(*out.shape), alpha=1 / self.n_workers)

        return bits_communicated

class Exact_AG_Reducer(Reducer):
    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0

        with self.timer("reduce.zero_mem", verbosity=2):
            for mem in memory_out:
                mem.zero_()

        with self.timer("reduce.build_lists", verbosity=2):
            list_in = grad_in
            list_out = grad_out

        with self.timer("reduce.gather", verbosity=2):
            if torch.distributed.is_available():
                n_workers = torch.distributed.get_world_size()
            else:
                n_workers = 1

            if n_workers == 1:
                for t_in, t_out in zip(list_in, list_out):
                    t_out[:] = t_in
                return 0

            with self.timer("reduce.gather.pack"):
                values_bfr = TensorBuffer(list_in)

            with self.timer("reduce.gather.allgather"):
                all_values_buffers = values_bfr.all_gather()
                bits_communicated += values_bfr.bits()

            with self.timer("reduce.gather.average"):
                for out in grad_out:
                    out.data[:] = 0.0

                # aggregate the results from all workers
                for values_buffer in all_values_buffers:
                    # initialize lists for unpacking
                    local_list_values = []
                    for t_in in list_in:
                        local_list_values.append(torch.empty_like(t_in))
                    values_bfr.buffer = values_buffer
                    values_bfr.unpack(local_list_values)
                    for values, out in zip(local_list_values, grad_out):
                        out.add_(values.view(*out.shape), alpha=1 / self.n_workers)

        return bits_communicated

class IntQuantReducer(Reducer):
    def __init__(self, random_seed, device, timer, alpha, beta, alpha0, rand_round, overflow_handling, int8, total_dim):
        super().__init__(random_seed, device, timer)
        self.alpha = alpha
        self.beta = beta
        self.alpha0 = alpha0
        self.rk = None
        self.rand_round = rand_round
        self.overflow_handling = overflow_handling
        self.int8 = int8
        self.total_dim = total_dim
        if self.rand_round:
            self.round_func = rand_int_quant
        else:
            self.round_func = torch.round
        if self.int8:
            self.comm_dtype = torch.int8
        else:
            self.comm_dtype = torch.int32

    def set_alpha(self, grad_norm_sq, n_workers):
        if self.rk is None:
            self.rk = grad_norm_sq
        else:
            self.rk *= self.beta
            self.rk += (1 - self.beta) * grad_norm_sq
        self.alpha = self.alpha0 * np.sqrt(self.total_dim) / (np.sqrt(2 * n_workers) * torch.sqrt(self.rk + 1e-16))

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        if torch.distributed.is_available():
            n_workers = torch.distributed.get_world_size()
        else:
            n_workers = 1

        bits_communicated = 0

        with self.timer("reduce.flatpack", verbosity=2):
            flat_values = TensorBuffer(grad_in)

        with self.timer("reduce.round", verbosity=2):
            rounded_buffer = self.round_func(self.alpha * flat_values.buffer)
            if self.overflow_handling:
                if self.int8:
                    max_int = 128.0
                else:
                    max_int = 2147483648.0
                rounded_buffer = torch.clamp(rounded_buffer, min=-max_int / n_workers, max=(max_int - 1) / n_workers)
            flat_values.buffer = rounded_buffer.to(dtype=self.comm_dtype)

        with self.timer("reduce.reduce", verbosity=2):
            flat_values.all_reduce()
            flat_values.buffer = flat_values.buffer.to(torch.float) / (n_workers * self.alpha)

        with self.timer("reduce.unflatpack", verbosity=2):
            flat_values.unpack(grad_out)

        with self.timer("reduce.reset_alpha", verbosity=2):
            grad_norm_sq = torch.sum(flat_values.buffer.detach().clone() ** 2)
            self.set_alpha(grad_norm_sq, n_workers)

        return bits_communicated


class HintQuantReducer(Reducer):
    def __init__(self, random_seed, device, timer, rand_round, overflow_handling, int8):
        super().__init__(random_seed, device, timer)
        self.rand_round = rand_round
        self.int8 = int8
        self.overflow_handling = overflow_handling
        if self.rand_round:
            self.round_func = rand_int_quant
        else:
            self.round_func = torch.round
        if self.int8:
            self.comm_dtype = torch.int8
        else:
            self.comm_dtype = torch.int32

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        if torch.distributed.is_available():
            n_workers = torch.distributed.get_world_size()
        else:
            n_workers = 1

        bits_communicated = 0

        with self.timer("reduce.flatpack", verbosity=2):
            flat_values = TensorBuffer(grad_in)

        with self.timer("reduce.profiling", verbosity=2):
            abs_max_val = torch.abs(torch.max(flat_values.buffer))
            torch.distributed.all_reduce(abs_max_val, op=torch.distributed.reduce_op.MAX)
            exponent = torch.floor(torch.log2(abs_max_val)).item()
            if self.int8:
                self.alpha = (2 ** 7 - 1.0) / (n_workers * (2 ** exponent))
            else:
                self.alpha = (2 ** 31 - 1.0) / (n_workers * (2 ** exponent))

        with self.timer("reduce.round", verbosity=2):
            rounded_buffer = self.round_func(self.alpha * flat_values.buffer)
            if self.overflow_handling:
                if self.int8:
                    max_int = 128.0
                else:
                    max_int = 2147483648.0
                rounded_buffer = torch.clamp(rounded_buffer, min=-max_int / n_workers, max=(max_int - 1) / n_workers)
            flat_values.buffer = rounded_buffer.to(dtype=self.comm_dtype)

        with self.timer("reduce.reduce", verbosity=2):
            flat_values.all_reduce()
            # compute nbits before converting it to float
            bits_communicated += flat_values.bits()
            flat_values.buffer = flat_values.buffer.to(torch.float) / (n_workers * self.alpha)

        with self.timer("reduce.unflatpack", verbosity=2):
            flat_values.unpack(grad_out)

        return bits_communicated


@torch.jit.script
def orthogonalize(matrix, eps=torch.tensor(1e-8)):
    n, m = matrix.shape
    for i in range(m):
        # Normalize the i'th column
        col = matrix[:, i: i + 1]
        col /= torch.sqrt(torch.sum(col ** 2)) + eps
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1:]
            # rest -= torch.matmul(col.t(), rest) * col
            rest -= torch.sum(col * rest, dim=0) * col


@torch.jit.script
def rand_int_quant(x):
    s = torch.sign(x)
    abs_vals = torch.abs(x)
    lower = torch.floor(abs_vals)
    upper = torch.ceil(abs_vals)
    prob = (abs_vals - lower) / (upper - lower + 3e-16)
    v = torch.bernoulli(prob)
    out = s * (lower + v)
    return out


def reduce_mean_list(
        device: torch.device, list_in: List[torch.Tensor], list_out: List[torch.Tensor], timer
):
    if torch.distributed.is_available():
        n_workers = torch.distributed.get_world_size()
    else:
        n_workers = 1

    if n_workers == 1:
        for t_in, t_out in zip(list_in, list_out):
            t_out[:] = t_in
        return 0

    with timer("reduce.mean.pack"):
        buffer = TensorBuffer(list_in)

    with timer("reduce.mean.allreduce"):
        buffer.all_reduce()
        buffer.buffer /= n_workers
        bits_communicated = buffer.bits()

    with timer("reduce.mean.unpack", verbosity=2):
        buffer.unpack(list_out)

    return bits_communicated


def n_bits(tensor):
    return 8 * tensor.nelement() * tensor.element_size()


class TensorBuffer():
    """
    Packs multiple tensors into one flat buffer for efficient
    intra-worker communication.
    """

    def __init__(self, tensors):
        indices = [0]
        for tensor in tensors:
            new_end = indices[-1] + tensor.nelement()
            indices.append(new_end)

        self._start_idx = indices[:-1]
        self._end_idx = indices[1:]
        self._tensors = tensors

        self.buffer = torch.cat([t.view(-1) for t in tensors])  # copies

    def __getitem__(self, index):
        return self.buffer[self._start_idx[index]: self._end_idx[index]].view(*self._tensors[index].shape)

    def __len__(self):
        return len(self._tensors)

    def pack(self, tensors=None):
        # Optional. init already does this.
        if tensors is None:
            tensors = self._tensors
        for tensor, entry in zip(tensors, self):
            entry[:] = tensor

    def unpack(self, tensors):
        for tensor, entry in zip(tensors, self):
            tensor[:] = entry

    def nelement(self):
        return self.buffer.nelement()

    def element_size(self):
        return self.buffer.element_size()

    def bits(self):
        return 8 * self.nelement() * self.element_size()

    def all_reduce(self, async_op=False):
        return torch.distributed.all_reduce(self.buffer, async_op=async_op)

    def all_gather(self, async_op=False):
        n_workers = torch.distributed.get_world_size() if torch.distributed.is_available() else 1
        buffers = [torch.empty_like(self.buffer) for i in range(n_workers)]
        handle = all_gather(buffers, self.buffer, async_op=async_op)
        if async_op:
            return buffers, handle
        else:
            return buffers


def all_reduce(*args, **kwargs):
    if torch.distributed.is_available() and torch.distributed.get_world_size() > 1:
        return torch.distributed.all_reduce(*args, **kwargs)


def all_gather(out_list, in_tensor, **kwargs):
    if torch.distributed.is_available() and torch.distributed.get_world_size() > 1:
        return torch.distributed.all_gather(out_list, in_tensor, **kwargs)
    else:
        assert len(out_list) == 1
        out_list[0].data = in_tensor


@torch.jit.script
def l2norm(x):
    return torch.sqrt(torch.sum(x ** 2))


def normalize_(tensor):
    """Divide by L2 norm. In place"""
    tensor /= l2norm(tensor)
