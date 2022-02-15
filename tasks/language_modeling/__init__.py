import os
from copy import deepcopy
from typing import Dict, Iterable, List

import torch

import tasks.language_data as data


from mean_accumulator import MeanAccumulator

from tasks.language_modeling.model import RNNModel

def batchify(data, bsz, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def get_batch(source, i, bptt, rank, bsz):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len, rank * bsz : (rank + 1) * bsz]
    target = source[i+1:i+1+seq_len, rank * bsz : (rank + 1) * bsz]
    return data, target.contiguous().view(-1)

class LanguageModelingTask:
    def __init__(self, device, timer, seed, batch_size, rnn_type):
        self._device = device
        self._timer = timer
        self._batch_size = batch_size
        self._test_batch_size = batch_size
        self._seed = seed
        self._epoch = 0
        self.rnn_type = rnn_type
        self.bptt = 30

        torch.random.manual_seed(self._seed)
        self.corpus, self.train_data, self.val_data, self.test_data = define_dataset(
            device=device,
            dataset_name="wikitext2",
            dataset_path="../data/wikitext-2",
            batch_size=self._batch_size,
            test_batch_size=self._test_batch_size,
        )

        self._model = self._create_model()
        self._criterion = torch.nn.CrossEntropyLoss().to(self._device)

        self.state = [parameter for parameter in self._model.parameters()]
        self.buffers = [buffer for buffer in self._model.buffers()]
        self.parameter_names = [name for (name, _) in self._model.named_parameters()]
        self._hidden_container = {"hidden": None}

    def update_epoch(self, epoch):
        self._epoch = epoch

    def reshuffle_train_data(self):
        new_idx = torch.randperm(self.train_data.size(1))
        self.train_data = self.train_data[:, new_idx]

    def init_hidden(self):
        self._hidden_container["hidden"] = self._model.init_hidden(self._batch_size)

    def train_iterator(self, i):
        rank = torch.distributed.get_rank() if torch.distributed.is_available() else 1
        return get_batch(self.train_data, i, self.bptt, rank, self._batch_size)

    def batch_loss_and_gradient(self, data, target, rnn_clip=0.4):
        self._zero_grad()
        with self._timer("batch.forward", float(self._epoch)):
            hidden = self._hidden_container["hidden"]
            prediction, hidden = self._model(data, hidden)
            hidden = self._model.repackage_hidden(hidden)
            self._hidden_container["hidden"] = hidden
            f = self._criterion(
                prediction.view(-1, self._model.ntokens), target.contiguous().view(-1)
            )
        with self._timer("batch.backward", float(self._epoch)):
            f.backward()
        with self._timer("batch.evaluate", float(self._epoch)):
            metrics = self.evaluate_prediction(prediction, target)
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), rnn_clip)
        df = [parameter.grad for parameter in self._model.parameters()]
        return f.detach(), df, metrics

    def evaluate_prediction(self, model_output, reference):
        """
        Compute a series of scalar loss values for a predicted batch and references
        """
        with torch.no_grad():
            cross_entropy = self._criterion(
                model_output.view(-1, self._model.ntokens), reference.contiguous().view(-1)
            )
            return {
                "cross_entropy": cross_entropy.detach(),
                "perplexity": torch.exp(cross_entropy).detach(),
            }

    def test(self, state_dict=None) -> float:
        """
        Compute the average loss on the test set.
        The task is completed as soon as the output is below self.target_test_loss.
        If the model has batch normalization or dropout, this will run in eval mode.
        """
        rank = torch.distributed.get_rank() if torch.distributed.is_available() else 1

        if state_dict:
            test_model = self._create_test_model(state_dict)
        else:
            test_model = self._model
            test_model.eval()

        hidden = test_model.init_hidden(self._test_batch_size)

        mean_metrics = MeanAccumulator()

        with torch.no_grad():
            for i in range(0, self.test_data.size(0) - 1, self.bptt):
                data, target = get_batch(self.test_data, i, bptt=self.bptt, rank=rank, bsz=self._test_batch_size)
                prediction, hidden = test_model(data, hidden)
                hidden = test_model.repackage_hidden(hidden)
                metrics = self.evaluate_prediction(prediction, target)
            mean_metrics.add(metrics)
        mean_metrics.reduce(device=self._device)  # Collect over workers

        test_model.train()
        return mean_metrics.value()

    def state_dict(self):
        """Dictionary containing the model state (buffers + tensors)"""
        return self._model.state_dict()

    def _create_model(self):
        """Create a PyTorch module for the model"""
        torch.random.manual_seed(self._seed)
        n_tokens = len(self.corpus.dictionary)
        model = define_model(n_tokens=n_tokens, emb_size=650, rnn_type=self.rnn_type)
        model.to(self._device)
        model.train()
        return model

    def _create_test_model(self, state_dict):
        test_model = deepcopy(self._model)
        test_model.load_state_dict(state_dict)
        test_model.eval()
        return test_model

    def _zero_grad(self):
        self._model.zero_grad()


def define_dataset(
    device,
    dataset_name,
    dataset_path,
    batch_size,
    test_batch_size,
):
    # create dataset.
    corpus = _get_dataset(dataset_name, dataset_path)

    n_workers = torch.distributed.get_world_size() if torch.distributed.is_available() else 1

    train_data = batchify(corpus.train, bsz=batch_size * n_workers, device=device)
    val_data = batchify(corpus.valid, bsz=batch_size * n_workers, device=device)
    test_data = batchify(corpus.test, bsz=test_batch_size * n_workers, device=device)

    return corpus, train_data, val_data, test_data


def define_model(n_tokens, emb_size, rnn_type, rnn_n_hidden=650, rnn_n_layers=3, rnn_tie_weights=True, drop_rate=0.4):
    # create model.
    model = RNNModel(
        rnn_type=rnn_type,
        ntoken=n_tokens,
        ninp=emb_size,
        nhid=rnn_n_hidden,
        nlayers=rnn_n_layers,
        tie_weights=rnn_tie_weights,
        dropout=drop_rate,
    )

    return model

def _get_dataset(name, datasets_path):
    if "wikitext2" in name:
        corpus = data.Corpus(datasets_path)
    else:
        raise ValueError("dataset not found")

    return corpus
