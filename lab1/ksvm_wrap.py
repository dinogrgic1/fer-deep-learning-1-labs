import torch
import numpy as np
import matplotlib.pyplot as plt
from data import *
from sklearn import svm
from pt_deep import PTDeep

class KSVMWrap:
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto', kernel='rbf'):
        self.model = svm.SVC(C=param_svm_c, kernel='rbf', gamma=param_svm_gamma, probability=True).fit(X, Y_)
    
    def predict(self, X):
        return self.model.predict(X)

    def get_scores(self, X):
        return self.model.predict_proba(X)

    def support(self):
        return self.model.support_

# Zadatak 6
if __name__ == '__main__':
    np.random.seed(100)
    _K = 6
    _C = 2
    _N = 10
    _NITER = 10000
    _DELTA = 1e-3
    _LAMBDA = 0
    _param_svm_c = 1
    _param_svm_gamma = 'auto'

    X, Y_ = sample_gmm_2d(_K, _C, _N)
    ksvm_wrap = KSVMWrap(X, Y_, _param_svm_c, _param_svm_gamma)
    
    graph_surface(ksvm_wrap.predict, get_box(X), 0.5, 1024, 1024) 
    graph_data(X, Y_, ksvm_wrap.predict(X), ksvm_wrap.support())

    accuracy, pr, M = eval_perf_multi(ksvm_wrap.predict(X), Y_)
    ap = eval_AP(Y_[np.argmax(ksvm_wrap.get_scores(X)).argsort()])
    metrics_print(accuracy, pr, ap)
    plt.show()

    # Comparrision to PTDeep
    Yoh_ = class_to_onehot(Y_)
    print(Yoh_)
    config = [2, 128, 128, _C]
    ptlr = PTDeep(config, torch.sigmoid)
    ptlr.train(torch.tensor(X, dtype=torch.float), torch.tensor(Yoh_), _NITER, _DELTA, _LAMBDA)
    Y = ptlr.eval(X)

    accuracy, pr, M = eval_perf_multi(Y, Y_)
    metrics_print(accuracy, pr, M)

    graph_surface(ptlr.eval, get_box(X), 0.5, 1024, 1024)
    graph_data(torch.tensor(X), torch.tensor(Y_), torch.tensor(Y))
    plt.title(f'Configuration {config} | λ = {_LAMBDA} | Δ = {_DELTA} | N = {_NITER}')
    plt.show()