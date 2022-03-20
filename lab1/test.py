from data import *
from matplotlib import pyplot as plt
import torch

from fcann2 import *
from pt_linreg import LinearRegression

def second():
    __K = 6
    __C = 2
    __N = 10
    __param_niter = 10000
    __param_delta = 0.05
    __param_lambda = 1e-3
    __nhidden = 5

    X, Y_ = sample_gmm_2d(__K, __C, __N)

    model = FCANN2()
    model.fcann2_train(X, Y_, __param_niter, __param_delta, __param_lambda, __nhidden)
    
    rect = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(model.fcann2_classify, rect, 0.5, 1024, 1024)
    graph_data(X, Y_, model.fcann2_classify(X), special=[])
    plt.show()

def third():
    lr = LinearRegression()

    X = torch.tensor([1, 2])
    Y_= torch.tensor([3, 5])
    _ = lr.train(X, Y_)

    rect = (torch.min(X, axis=0), torch.max(X, axis=0))
    graph_surface(lr.classify, rect)
    graph_data(X, Y_, lr.classify(X), special=[])
    plt.show()


if __name__ == '__main__':
    np.random.seed(100)
    third()
