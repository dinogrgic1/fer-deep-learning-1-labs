import torch
import numpy as np
import matplotlib.pyplot as plt
from data import *

class PTLogreg(torch.nn.Module):
    def __init__(self, D: int, C: int) -> None:
        """Arguments:
            - D: dimensions of each datapoint 
            - C: number of classes
        """
        
        # Module msut be frist initiazlied
        super().__init__()
        self.D = D
        self.C = C
        self.W = torch.nn.Parameter(torch.randn((D, C)), requires_grad=True)
        self.B = torch.nn.Parameter(torch.zeros(C), requires_grad=True)

    def forward(self, X: torch.tensor) -> torch.tensor:
        output = X.mm(self.W) + self.B
        return torch.softmax(output, dim=1)

    def get_loss(self, X: torch.tensor, Yoh_: torch.tensor, param_lambdba: float) -> torch.tensor:
        probs = self.forward(X)
        return -torch.mean(torch.sum(torch.log(probs) * Yoh_, dim=1)) + torch.norm(self.W) * param_lambdba


    def train(self, X: torch.tensor, Yoh_: torch.tensor, param_niter: int, param_delta: float, param_lambdba = 0.) -> None:
        """Arguments:
            - X: model inputs [NxD], type: torch.Tensor
            - Yoh_: ground truth [NxC], type: torch.Tensor
            - param_niter:  number of training iterations
            - param_delta: learning rate
        """
        optimizer = torch.optim.SGD(params=self.parameters(), lr = param_delta, weight_decay = param_lambdba)
        for i in range(param_niter + 1):
            loss = self.get_loss(X, Yoh_, param_lambdba)
            loss.backward()
            optimizer.step()      

            if i % 10 == 0:
                print(f'step: {i}\tloss: {loss:.6f}')
            optimizer.zero_grad()
            

    def eval(self, X: np.array) -> np.array:
        """Arguments:
            - model: type: PTLogreg
            - X: actual datapoints [NxD], type: np.array
            Returns: predicted class probabilites [NxC], type: np.array
        """
        probs = self.forward(torch.tensor(X, dtype=torch.float32))
        probs = probs.detach().numpy()
        return np.argmax(probs, axis=1)

# Zadatak 4
if __name__ == "__main__":
    np.random.seed(100)
    
    _C = 3
    _NITER = 1000
    _LAMBDAS = [1e-3, 1e-2, 0, 0.5, 1]
    _DELTA = 1e-2

    X, Y_ = sample_gmm_2d(3, _C, 10)
    Yoh_ = class_to_onehot(Y_)

    for _L in _LAMBDAS:
        ptlr = PTLogreg(X.shape[1], Yoh_.shape[1])
        ptlr.train(torch.tensor(X, dtype=torch.float), torch.tensor(Yoh_), _NITER, _DELTA, _L)
        Y = ptlr.eval(X)

        accuracy, pr, M = eval_perf_multi(Y, Y_)
        metrics_print(accuracy, pr, None)

        graph_surface(ptlr.eval, get_box(X), 0.5, 1024, 1024)
        graph_data(torch.tensor(X), torch.tensor(Y_), torch.tensor(Y))
        plt.title(f'λ = {_L} | Δ = {_DELTA} | N = {_NITER}')
        plt.show()