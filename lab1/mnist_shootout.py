import torch
import torchvision
import json
from pt_deep import PTDeep
import torch
import matplotlib.pyplot as plt
import time
from data import class_to_onehot, eval_perf_multi, metrics_print

dataset_root = 'dataset/'  # change this to your preference
traindata_root = 'train_data'
mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)

x_train, y_train = mnist_train.data, mnist_train.targets
x_test, y_test = mnist_test.data, mnist_test.targets
x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

y_train_oh = class_to_onehot(y_train)
x_train = torch.tensor(x_train, dtype=torch.float).clone().detach().reshape(-1, 784)
x_test = torch.tensor(x_test, dtype=torch.float).clone().detach().reshape(-1, 784)

__LAMBDA = 1e-2
__DELTA = 0.1
__COUNT = 3000

CONFIGS = [[784, 10], [784, 100, 10]] # NO-GPU
#CONFIGS = [[784, 10], [784, 100, 10], [784, 100, 100, 10], [784, 100, 100, 100, 10]]
for config in CONFIGS:
    ptlr = PTDeep(config, torch.relu)
    losses = ptlr.train(x_train, torch.tensor(y_train_oh), __COUNT, __LAMBDA, __DELTA, save_loss=True, optimizer=torch.optim.Adam)

    Y = ptlr.eval(x_test.detach().numpy())
    accuracy, pr, M = eval_perf_multi(Y, y_test)
    metrics_print(accuracy, pr, None)

    config_str = ('-').join(map(str, config))
    run = {"config": config_str, "lambda": __LAMBDA, "delta": __DELTA, "niter": __COUNT, "accuracy": accuracy, "losses": losses}
    timestr = time.strftime("%Y%m%d-%H%M%S")
    with open(f'{traindata_root}/{config_str}_{timestr}_data.json', 'w+') as outfile:
        json.dump(run, outfile)


        