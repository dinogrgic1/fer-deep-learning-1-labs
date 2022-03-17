import torch
import numpy as np

def fcann2_train(X, Y_, niter, delta, llambda, nhidden):
    X = torch.tensor(X, dtype=torch.double)
    Y_ = torch.tensor(Y_, dtype=torch.double)

    # 2 x 5
    W1 = torch.randn(2, nhidden, dtype=torch.double)
    # 1 x 5
    b1 = torch.zeros(1, nhidden, dtype=torch.double)
    # (N x 2) x (2 x 5) -> N X 5
    s1 = X @ W1 + b1
    h1 = torch.relu(s1)

    # 5 x 2
    W2 = torch.randn(nhidden, 2, dtype=torch.double)
    b2 = torch.randn(len(X), 2, dtype=torch.double)
    s2 = h1 @ W2 + b2

    loss = torch.nn.CrossEntropyLoss()
    output = loss(s2, Y_)
    print(output)


def fcann2_classify():
    return 0