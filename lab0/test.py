from distutils.log import Log
import numpy as np
import matplotlib.pyplot as plt

from data import sample_gauss_2d
from binlogreg import LogisticRegresion

if __name__=="__main__":
    np.random.seed(100)

    # get the training dataset
    X, Y_ = sample_gauss_2d(2, 100)

    # train the model
    lr = LogisticRegresion()
    w,b = lr.binlogreg_train(X, Y_)

    # # evaluate the model on the training dataset
    # probs = binlogreg_classify(X, w,b)
    # Y = # TODO

    # # report performance
    # accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    # AP = data.eval_AP(Y_[probs.argsort()])
    # print (accuracy, recall, precision, AP)