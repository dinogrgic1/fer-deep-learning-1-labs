import torch
import torchvision
from pt_deep import PTDeep
import torch
import numpy as np
import matplotlib.pyplot as plt
from data import sample_gmm_2d, class_to_onehot, eval_perf_multi, graph_surface, graph_data, metrics_print

dataset_root = 'data/mnist'  # change this to your preference
mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)

x_train, y_train = mnist_train.data, mnist_train.targets
x_test, y_test = mnist_test.data, mnist_test.targets
x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

y_train_oh = class_to_onehot(y_train)
x_train = torch.tensor(x_train, dtype=torch.float).clone().detach().reshape(-1, 784)
x_test = torch.tensor(x_test, dtype=torch.float).clone().detach().reshape(-1, 784)

CONFIGS = [[784, 10]]
#CONFIGS = [[784, 10], [784, 100, 10], [784, 100, 100, 10], [784, 100, 100, 100, 10]]
for config in CONFIGS:
    ptlr = PTDeep(config, torch.relu)
    ptlr.train(x_train, torch.tensor(y_train_oh), 3000, 0.5, 1e-2)
    Y = ptlr.eval(x_test.detach().numpy())

    print(Y)
    print(y_test)
    accuracy, pr, M = eval_perf_multi(Y, y_test)
    metrics_print(accuracy, pr, None)
        