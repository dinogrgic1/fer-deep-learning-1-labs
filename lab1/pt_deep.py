import torch
import numpy as np
import matplotlib.pyplot as plt
from data import sample_gmm_2d, class_to_onehot, eval_perf_multi, graph_surface, graph_data

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
            w_s.append(torch.nn.Parameter(torch.randn((config[i], config[i + 1])), requires_grad=True))
            b_s.append(torch.nn.Parameter(torch.zeros(config[i + 1]), requires_grad=True))

        self.weights = torch.nn.ParameterList(w_s)
        self.biases = torch.nn.ParameterList(b_s)
        self.activation = activ_function
        

    def forward(self, X: torch.tensor) -> torch.tensor:
        output = X
        for layer in range(len(self.weights)):
            output = output.mm(self.weights[layer]) + self.biases[layer]
            output = self.activation(output)
        return torch.softmax(output, dim=1)

    def get_loss(self, X: torch.tensor, Yoh_: torch.tensor, param_lambdba: float) -> torch.tensor:
        probs = self.forward(X)
        return -torch.mean(torch.sum(torch.log(probs) * Yoh_, dim=1))


    def train(self, X: torch.tensor, Yoh_: torch.tensor, param_niter: int, param_delta: float, param_lambdba = 0.) -> None:
        """Arguments:
            - X: model inputs [NxD], type: torch.Tensor
            - Yoh_: ground truth [NxC], type: torch.Tensor
            - param_niter:  number of training iterations
            - param_delta: learning rate
        """
        optimizer = torch.optim.SGD(params=self.parameters(), lr = param_delta)
        for i in range(param_niter + 1):
            loss = self.get_loss(X, Yoh_, param_lambdba)
            loss.backward()
            optimizer.step()      

            if i % 10 == 0:
                print(f'step: {i}\tloss: {loss:.6f}')
            optimizer.zero_grad()
        pass
            

    def eval(self, X: np.array) -> np.array:
        probs = self.forward(torch.tensor(X, dtype=torch.float32))
        probs = probs.detach().numpy()
        return np.argmax(probs, axis=1)

if __name__ == "__main__":
    np.random.seed(100)
    
    CONFIGS = [[2, 2], [2, 10, 2], [2, 10, 10, 2]]
    X, Y_ = sample_gmm_2d(4, 2, 40)
    Yoh_ = class_to_onehot(Y_)

    for config in CONFIGS:
        ptlr = PTDeep(config, torch.sigmoid)
        ptlr.train(torch.tensor(X, dtype=torch.float), torch.tensor(Yoh_), 1000, 1e-2, 0.5)
        Y = ptlr.eval(X)

        accuracy, recall, precision = eval_perf_multi(Y, Y_)
        print(f'accuracy: {accuracy:.4f}')
        for i in range(2):
            print(f'class #{i} recall: {recall[i]}\tprecision: {precision[i]}')

        box = (np.min(X, axis=0) - 0.5, np.max(X, axis=0) + 0.5)
        graph_surface(ptlr.eval, box, 0.5, 1024, 1024)
        graph_data(torch.tensor(X), torch.tensor(Y_), torch.tensor(Y))
        plt.show()
        