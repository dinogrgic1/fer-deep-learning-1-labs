import numpy as np
import matplotlib.pyplot as plt
from data import sample_gmm_2d, graph_surface, graph_data, get_box, eval_perf_multi, eval_AP, metrics_print
from sklearn import svm

class KSVMWrap:
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        self.model = svm.SVC(C=param_svm_c, gamma=param_svm_gamma, probability=True).fit(X, Y_)
    
    def predict(self, X):
        return self.model.predict(X)

    def get_scores(self, X):
        return self.model.predict_proba(X)

    def support(self):
        return self.model.support_

if __name__ == '__main__':
    np.random.seed(100)
    K=6
    C=2
    N=10
    param_svm_c=1
    param_svm_gamma='auto'

    X, Y_ = sample_gmm_2d(K, C, N)
    ksvm_wrap = KSVMWrap(X, Y_, param_svm_c, param_svm_gamma)
    
    graph_surface(ksvm_wrap.predict, get_box(X), 0.5, 1024, 1024) 
    graph_data(X, Y_, ksvm_wrap.predict(X), ksvm_wrap.support())

    accuracy, pr, M = eval_perf_multi(ksvm_wrap.predict(X), Y_)
    ap = eval_AP(Y_[np.argmax(ksvm_wrap.get_scores(X)).argsort()])
    metrics_print(accuracy, pr, ap)
    plt.show()