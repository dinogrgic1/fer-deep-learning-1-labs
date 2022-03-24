import torch
import numpy as np
import matplotlib.pyplot as plt
from data import *

class PTDeep(torch.nn.Module):
    def __init__(self, config: np.array, activ_function) -> None:
        """Arguments:
            - config: array configuration
            - activ_function: activation function between layers
        """

        super().__init__()
        w_s = []
        b_s = []
        for i in range(0, len(config)-1):
            w_s.append(torch.nn.Parameter(torch.randn((config[i], config[i + 1]),  dtype=torch.float), requires_grad=True))
            b_s.append(torch.nn.Parameter(torch.zeros(config[i + 1], dtype=torch.float), requires_grad=True))

        self.weights = torch.nn.ParameterList(w_s)
        self.biases = torch.nn.ParameterList(b_s)
        self.activation = activ_function
        self.optimizer = None

    def forward(self, X: torch.tensor) -> torch.tensor:
        for layer in range(len(self.weights)):
            X = X.mm(self.weights[layer]) + self.biases[layer]
            X = self.activation(X)
        return torch.softmax(X, dim=1)

    def get_loss(self, X: torch.tensor, Yoh_: torch.tensor, param_lambdba: float) -> torch.tensor:
        probs = self.forward(X)
        return -torch.mean(torch.sum(torch.log(probs + 1e-13) * Yoh_, dim=1))


    def train(self, X: torch.tensor, Yoh_: torch.tensor, param_niter: int, param_delta: float, param_lambdba = 0., save_loss=False, optimizer=None) -> None:
        """Arguments:
            - X: model inputs [NxD], type: torch.Tensor
            - Yoh_: ground truth [NxC], type: torch.Tensor
            - param_niter:  number of training iterations
            - param_delta: learning rate
        """
        losses = []

        if optimizer != None:
            optimizer = optimizer(self.parameters(), lr = param_delta, weight_decay = param_lambdba)
        else:
            optimizer = torch.optim.SGD(params=self.parameters(), lr = param_delta, weight_decay = param_lambdba)
        for i in range(param_niter + 1):
            loss = self.get_loss(X, Yoh_, param_lambdba)
            if save_loss == True:
                losses.append(str(loss.detach().numpy()))
            loss.backward()
            optimizer.step()      

            if i % 10 == 0:
                print(f'step: {i}\tloss: {loss:.6f}')
            optimizer.zero_grad()
        return losses
            
    def eval(self, X: np.array) -> np.array:
        probs = self.forward(torch.tensor(X, dtype=torch.float32))
        probs = probs.detach().numpy()
        return np.argmax(probs, axis=1)

    def count_params(self):
        layer = []
        for param in self.named_parameters():
            layer.append(f'{param[0]}: {param[1].shape}')
        n_parametars = sum(param.numel() for param in self.parameters())
        return layer, n_parametars

# Zadatak 5
if __name__ == "__main__":
    np.random.seed(100)
    
    _CONFIGS = [[2, 2], [2, 10, 2], [2, 10, 10, 2], [2, 10, 10, 3]]
    _LAMBDA = 1e-3
    _DELTA = 1e-2
    _NITER = 5000

    for config in _CONFIGS:
        X, Y_ = sample_gmm_2d(6, config[-1], 20)
        Yoh_ = class_to_onehot(Y_)

        ptlr = PTDeep(config, torch.sigmoid)
        ptlr.train(torch.tensor(X, dtype=torch.float), torch.tensor(Yoh_), _NITER, _DELTA, _LAMBDA)
        Y = ptlr.eval(X)

        accuracy, pr, M = eval_perf_multi(Y, Y_)
        metrics_print(accuracy, pr, M)
        layers, num_parameters = ptlr.count_params()
        print(f'Deep model has {num_parameters} parameters')
        print(f'Layers of deep model are: ')
        for layer in layers:
            print(f'\t{layer}')

        graph_surface(ptlr.eval, get_box(X), 0.5, 1024, 1024)
        graph_data(torch.tensor(X), torch.tensor(Y_), torch.tensor(Y))
        plt.title(f'Configuration {config} | λ = {_LAMBDA} | Δ = {_DELTA} | N = {_NITER}')
        plt.show()
        