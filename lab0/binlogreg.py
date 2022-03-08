import decimal
import numpy as np

def stable_softmax(x):
    exp_x_shifted = np.exp(x - np.max(x))
    probs = exp_x_shifted / np.sum(exp_x_shifted)
    return probs

class LogisticRegresion:
    param_niter = 500
    param_delta = 0.0005

    
    def binlogreg_train(self, X : np.array, Y_ : np.array):
        w = np.random.randn(len(X[0]))
        b = 0
        N = len(X)

        for i in range(self.param_niter):
            scores = np.dot(X, w) + b
            probs = stable_softmax(scores)
            print(probs)
            loss  = (-1 / N) * np.sum(np.log(probs))
            
            if i % 10 == 0:
                print(f"iteration {i}: loss {loss}")

            dL_dscores = probs - Y_
            print(dL_dscores)
            
            grad_w = (1 / N) * np.dot(dL_dscores, X)
            grad_b = np.sum(dL_dscores, axis=1)

            # poboljšani parametri
            w += -self.param_delta * grad_w
            b += -self.param_delta * grad_b
        return w, b