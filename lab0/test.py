from distutils.log import Log
import numpy as np
import matplotlib.pyplot as plt

from data import Random2DGaussian, sample_gauss_2d, eval_perf_binary, eval_AP
from binlogreg import LogisticRegresion

if __name__=="__main__":
    np.random.seed(100)

    data = Random2DGaussian()
    X, Y_ = sample_gauss_2d(2, 100)

    lr = LogisticRegresion()
    w,b = lr.binlogreg_train(X, Y_)
    probs = lr.binlogreg_classify(X, w,b)
    Y = np.where(probs >= 0.5, 1, 0)
    
    accuracy, recall, precision = eval_perf_binary(Y, Y_)
    AP = eval_AP(Y_[probs.argsort()])
    print(accuracy, recall, precision, AP)