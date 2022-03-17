import numpy as np

def stable_softmax(x):
    return np.exp(x) / (1 + np.exp(x))

class LogisticRegresion:
    param_niter = 500
    param_delta = 0.0005

    def binlogreg_train(self, X : np.array, Y_ : np.array):
        w = np.random.randn(len(X[0]))
        b = 0
        N = len(X)

        for i in range(self.param_niter + 1):
            scores = np.dot(X, w) + b
            probs = np.exp(scores) / (1 + np.exp(scores))

            loss = np.sum([Y_, np.log(probs)], where=np.where(Y_ == 1, True, False), axis=1)[1]
            loss += np.sum([Y_, np.log(1 - probs)], where=np.where(Y_ == 0, True, False), axis=1)[1]
            loss *= -(1 / N)

            if i % 10 == 0:
                print(f"iteration {i}: loss {loss}")

            dL_dscores = []
            for prob, y in zip(probs, Y_):
                if y == 1:
                    dL_dscores.append(prob - 1)
                else:
                    dL_dscores.append(prob)
            dL_dscores = np.array(dL_dscores)

            grad_w = (1 / N) * np.dot(dL_dscores, X)
            grad_b = np.sum(dL_dscores, axis=0)

            w += -self.param_delta * grad_w
            b += -self.param_delta * grad_b
        return w, b
    
    def binlogreg_classify(self, X : np.array, w : np.array, b: np.array):
        scores = np.dot(X, w) + b
        return np.exp(scores) / (1 + np.exp(scores))

