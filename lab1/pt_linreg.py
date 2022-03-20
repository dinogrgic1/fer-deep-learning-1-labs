import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LinearRegression:

    ntimes = 100
    X = None
    Y_ = None
    a = None
    b = None

    def train(self, X, Y_):
        self.X = X
        self.Y_ = Y_
        self.a = torch.randn(1, requires_grad=True)
        self.b = torch.randn(1, requires_grad=True)

        optimizer = optim.SGD([self.a, self.b], lr=0.1)
        for i in range(self.ntimes):
            Y_ = self.a * X + self.b
            diff = (self.Y_-Y_)
            loss = torch.sum(diff**2)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 10 == 0:
                print(f'step: {i}, loss:{loss}, Y_:{Y_}, a:{self.a}, b {self.b}')
        
        return Y_

    def classify(self, X):
        X = torch.tensor(X)
        return self.a * X + self.b
