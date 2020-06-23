#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Generate function and model benchmarks

To Run:
$ python benchmark.py

# only function benchmarks
$ python benchmark.py --only-functions
$ python benchmark.py --only-functions --world-size 2

# benchmark functions and all models
$ python benchmark.py --advanced-models

# save benchmarks to csv
$ python benchmark.py -p ~/Downloads/
"""

import argparse
import functools
import os
import timeit
from collections import namedtuple

import crypten
import crypten.communicator as comm
import numpy as np
import pandas as pd
import torch
from examples import multiprocess_launcher


try:
    from . import data
    from . import models
except ImportError:
    # direct import if relative fails
    import data
    import models

import syft
from syft.frameworks import crypten as syft_crypten
from syft.workers.node_client import NodeClient


torch_hook = syft.TorchHook(torch)

Runtime = namedtuple("Runtime", "mid q1 q3")


# alice = syft.VirtualWorker(torch_hook, "alice")
# bob = syft.VirtualWorker(torch_hook, "bob")
alice = NodeClient(torch_hook, "ws://localhost:3000")
bob = NodeClient(torch_hook, "ws://localhost:3001")
WORKERS = [alice, bob]
IP = "127.0.0.1"


def time_me(func=None, n_loops=10):
    """Decorator returning average runtime in seconds over n_loops

    Args:
        func (function): invoked with given args / kwargs
        n_loops (int): number of times to invoke function for timing

    Returns: tuple of (time in seconds, inner quartile range, function return value).
    """
    if func is None:
        return functools.partial(time_me, n_loops=n_loops)

    @functools.wraps(func)
    def timing_wrapper(*args, **kwargs):
        return_val = func(*args, **kwargs)
        times = []
        for _ in range(n_loops):
            start = timeit.default_timer()
            func(*args, **kwargs)
            times.append(timeit.default_timer() - start)
        mid_runtime = np.quantile(times, 0.5)
        q1_runtime = np.quantile(times, 0.25)
        q3_runtime = np.quantile(times, 0.75)
        runtime = Runtime(mid_runtime, q1_runtime, q3_runtime)
        return runtime, return_val

    return timing_wrapper


class ModelBenchmarks:
    """Benchmarks runtime and accuracy of crypten models

    Models are benchmarked on synthetically generated
    Gaussian clusters for binary classification. Resnet18 is
    benchmarks use image data.

    Args:
        n_samples (int): number of samples for Gaussian cluster model training
        n_features (int): number of features for the Gaussian clusters.
        epochs (int): number of training epochs
        lr_rate (float): learning rate.
    """

    def __init__(self, advanced_models=False):
        self.df = None

        self.models = models.MODELS
        if not advanced_models:
            self.remove_advanced_models()

    def __repr__(self):
        if self.df is not None:
            return self.df.to_string(index=False, justify="left")
        return "No Model Benchmarks"

    def remove_advanced_models(self):
        """Removes advanced models from instance"""
        self.models = list(filter(lambda x: not x.advanced, self.models))

    @time_me(n_loops=3)
    def train(self, model, x, y, epochs, lr, loss):
        """Trains PyTorch model

        Args:
            model (PyTorch model): model to be trained
            x (torch.tensor): inputs
            y (torch.tensor): targets
            epochs (int): number of training epochs
            lr (float): learning rate
            loss (str): type of loss to use for training

        Returns:
            model with update weights
        """
        assert isinstance(model, torch.nn.Module), "must be a PyTorch model"
        criterion = getattr(torch.nn, loss)()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        for _ in range(epochs):
            model.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        return model

    @time_me(n_loops=3)
    def train_crypten(self, model, x, y, epochs, lr, loss):
        """Trains crypten encrypted model

        Args:
            model (PyTorch model): model to be trained
            x (crypten.tensor): inputs
            y (crypten.tensor): targets
            epochs (int): number of training epochs
            lr (float): learning rate
            loss (str): type of loss to use for training

        Returns:
            model with update weights
        """
        assert isinstance(model, torch.nn.Module), "must be a PyTorch model"
        # criterion = getattr(crypten.nn, loss)()

        # for _ in range(epochs):
        #     model.zero_grad()
        #     output = model(x)
        #     loss = criterion(output, y)
        #     loss.backward()
        #     model.update_parameters(lr)

        epochs = torch.tensor(epochs)
        epochs.tag("epochs")
        ptrs = []
        for worker in WORKERS:
            ptrs.append(epochs.send(worker))

        @syft_crypten.context.run_multiworkers(WORKERS, master_addr=IP, model=model, dummy_input=x)
        def training():
            rank = crypten.communicator.get().get_rank()
            worker = syft.frameworks.crypten.get_worker_from_rank(rank)
            epochs = worker.search("epochs")[0]

            x = crypten.load("x", 0)
            y = crypten.load("y", 0)

            model.encrypt()
            criterion = crypten.nn.BCELoss()
            for _ in range(epochs):
                model.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                model.update_parameters(0.1)

            model.decrypt()
            return model

        result = training()
        # remove epochs
        for ptr in ptrs:
            _ = ptr.get()
        model = result[0]
        return model

    def time_training(self):
        """Returns training time per epoch for plain text and CrypTen"""
        runtimes = []
        runtimes_enc = []

        for model in self.models:
            x, y = model.data.x, model.data.y
            model_plain = model.plain()
            runtime, _ = self.train(model_plain, x, y, 1, model.lr, model.loss)
            runtimes.append(runtime)

            x_enc, y_enc = model.data.x_enc, model.data.y_enc
            # prepare data
            x = x_enc.get_plain_text()
            x.tag("x")
            y = y_enc.get_plain_text()
            y.tag("y")
            x_ptr = x.send(WORKERS[0])
            y_ptr = y.send(WORKERS[0])

            runtime_enc, _ = self.train_crypten(
                model_plain, x, y, 1, model.lr, model.loss
            )
            runtimes_enc.append(runtime_enc)

            # remove remote tensors
            _ = x_ptr.get()
            _ = y_ptr.get()

        return runtimes, runtimes_enc

    @time_me(n_loops=3)
    def predict_plain(self, model, x):
        y = model(x)
        return y

    @time_me(n_loops=3)
    def predict_enc(self, model, x):
        @syft_crypten.context.run_multiworkers(WORKERS, master_addr=IP, model=model, dummy_input=x)
        def pred():
            model.encrypt()
            x = crypten.load("x", 0)
            y = model(x)
            return y.get_plain_text()
        result = pred()
        y = result[0]
        return y

    def time_inference(self):
        """Returns inference time for plain text and CrypTen"""
        runtimes = []
        runtimes_enc = []

        for model in self.models:
            model_plain = model.plain()
            runtime, _ = self.predict_plain(model_plain, model.data.x)
            runtimes.append(runtime)

            x = model.data.x
            x.tag("x")
            ptr = x.send(WORKERS[0])
            
            runtime_enc, _ = self.predict_enc(model_plain, model.data.x)
            runtimes_enc.append(runtime_enc)

            _ = ptr.get()

        return runtimes, runtimes_enc

    @staticmethod
    def calc_accuracy(output, y, threshold=0.5):
        """Computes percent accuracy

        Args:
            output (torch.tensor): model output
            y (torch.tensor): true label
            threshold (float): classification threshold

        Returns (float): percent accuracy
        """
        predicted = (output > threshold).float()
        correct = (predicted == y).sum().float()
        accuracy = float((correct / y.shape[0]).numpy())
        return accuracy

    def evaluate(self):
        """Evaluates accuracy of crypten versus plain text models"""
        accuracies, accuracies_crypten = [], []

        for model in self.models:
            model_plain = model.plain()
            x, y = model.data.x, model.data.y
            _, model_plain = self.train(
                model_plain, x, y, model.epochs, model.lr, model.loss
            )

            x_test, y_test = model.data.x_test, model.data.y_test
            accuracy = ModelBenchmarks.calc_accuracy(model_plain(x_test), y_test)
            accuracies.append(accuracy)

            x_enc, y_enc = model.data.x_enc, model.data.y_enc
            # prepare data
            x = x_enc.get_plain_text()
            x.tag("x")
            y = y_enc.get_plain_text()
            y.tag("y")
            x_ptr = x.send(WORKERS[0])
            y_ptr = y.send(WORKERS[0])

            _, model_crypten = self.train_crypten(
                model_plain, x, y, model.epochs, model.lr, model.loss
            )

            x_enc, y_enc = model.data.x_enc, model.data.y_enc
            _, model_crypten = self.train_crypten(
                model_plain, x, y, model.epochs, model.lr, model.loss
            )
            x_test_enc = model.data.x_test_enc

            model_crypten.encrypt()
            output = model_crypten(x_test_enc).get_plain_text()
            accuracy = ModelBenchmarks.calc_accuracy(output, y_test)
            accuracies_crypten.append(accuracy)

        return accuracies, accuracies_crypten

    def save(self, path):
        self.df.to_csv(os.path.join(path, "model_benchmarks.csv"), index=False)

    def run(self):
        """Runs and stores benchmarks in self.df"""
        training_runtimes, training_runtimes_enc = self.time_training()
        inference_runtimes, inference_runtimes_enc = self.time_inference()
        accuracies, accuracies_crypten = self.evaluate()
        model_names = [model.name for model in self.models]

        training_times_both = training_runtimes + training_runtimes_enc
        inference_times_both = inference_runtimes + inference_runtimes_enc

        half_n_rows = len(training_runtimes)
        self.df = pd.DataFrame.from_dict(
            {
                "model": model_names + model_names,
                "seconds per epoch": [t.mid for t in training_times_both],
                "seconds per epoch q1": [t.q1 for t in training_times_both],
                "seconds per epoch q3": [t.q3 for t in training_times_both],
                "inference time": [t.mid for t in inference_times_both],
                "inference time q1": [t.q1 for t in inference_times_both],
                "inference time q3": [t.q3 for t in inference_times_both],
                "is plain text": [True] * half_n_rows + [False] * half_n_rows,
                "accuracy": accuracies + accuracies_crypten,
            }
        )
        self.df = self.df.sort_values(by="model")


def get_args():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(description="Benchmark Functions")
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        required=False,
        default=None,
        help="path to save function benchmarks",
    )
    parser.add_argument(
        "--advanced-models",
        required=False,
        default=False,
        action="store_true",
        help="run advanced model (resnet, transformer, etc.) benchmarks",
    )
    args = parser.parse_args()
    return args


def main():
    """Runs benchmarks and saves if path is provided"""
    args = get_args()
    benchmarks = [
        ModelBenchmarks(advanced_models=args.advanced_models),
    ]

    pd.set_option("display.precision", 3)
    for benchmark in benchmarks:
        benchmark.run()
        print(benchmark)
        if args.path:
            benchmark.save(args.path)


if __name__ == "__main__":
    main()
