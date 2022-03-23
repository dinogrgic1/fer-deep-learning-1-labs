import numpy as np
from data import *
from matplotlib import pyplot as plt

def relu(arr):
    arr[arr < 0] = 0
    return arr

def softmax(arr):
    arr = np.array(arr)
    max = np.max(arr, axis=1, keepdims=True)
    e_x = np.exp(arr - max)
    sum = np.sum(e_x, axis=1, keepdims=True)
    return  e_x / sum 

def one_hot(arr):
    one_hots = np.zeros((arr.size, arr.max()+1))
    one_hots[np.arange(arr.size), arr] = 1
    return one_hots

class FCANN2:
    N = None
    W1 = None
    W2 = None
    b1 = None
    b2 = None

    def loss(self, probs, Y_, llambda):
        probs_choice = np.take_along_axis(probs, Y_[:,None],axis=1)[:,0]
        net_loss = -np.mean(np.log(probs_choice))
        regularization_loss = llambda * np.sum([np.sum(np.square(x)) for x in (self.W1, self.W2)])
        return net_loss + regularization_loss
    
    def fcann2_train(self, X, Y_, niter, delta, llambda, nhidden):
        self.N = len(X)
        self.W1 = np.random.normal(scale=np.reciprocal(np.mean((2, nhidden))), size=(2, nhidden))
        self.b1 = np.zeros((1, nhidden))
        self.W2 = np.random.normal(scale=np.reciprocal(np.mean((nhidden, 2))), size=(nhidden, 2))
        self.b2 = np.zeros((1, 2))  

        for i in range(1, niter + 1):    
            s1 = X @ self.W1 + self.b1
            h1 = relu(s1)

            s2 = h1 @ self.W2 + self.b2
            probs = softmax(s2)

            loss = self.loss(probs, Y_, llambda)
    
            if i % 10 == 0:
                print(f'itteration: {i}, loss: {loss}')

            Gs2 = probs - one_hot(Y_)
            Gs2 = Gs2 / self.N
            grad_W2 = np.transpose(h1) @ Gs2
            grad_b2 = np.sum(Gs2, axis=0)

            Gs1 = Gs2 @ np.transpose(self.W2)
            Gs1[h1 <= 0.] = 0.
            grad_W1 = np.mean(np.transpose(X) @ Gs1, axis=0)
            grad_b1 = np.sum(Gs1, axis=0)

            self.W2 -= delta * grad_W2
            self.b2 -= delta * grad_b2
            self.W1 -= delta * grad_W1
            self.b1 -= delta * grad_b1

    def fcann2_classify(self, X):
        s1 = X @ self.W1 + self.b1
        h1 = relu(s1)

        s2 = h1 @ self.W2 + self.b2
        probs = softmax(s2)
        return np.argmax(probs, axis=1)

# Zadatak 2
if __name__ == '__main__':
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
    
    graph_surface(model.fcann2_classify, get_box(X), 0.5, 1024, 1024)
    graph_data(X, Y_, model.fcann2_classify(X), special=[])
    accuracy, pr, M = eval_perf_multi(model.fcann2_classify(X), Y_)
    metrics_print(accuracy, pr, None)

    plt.show()