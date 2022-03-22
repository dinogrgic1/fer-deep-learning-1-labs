import torch
import numpy as np
import matplotlib.pyplot as plt
from data import sample_gmm_2d, class_to_onehot, eval_perf_multi, graph_surface, graph_data

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
        self.W = torch.nn.Parameter(torch.randn(D, C, dtype=torch.double), requires_grad=True)
        self.B = torch.nn.Parameter(torch.zeros(C), requires_grad=True)
        
        # inicijalizirati parametre (koristite nn.Parameter):
        # imena mogu biti self.W, self.b
        # ...

    def forward(self, X: torch.tensor) -> torch.tensor:
        output = torch.mm(X, self.W) + self.B
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
        optimizer = torch.optim.SGD(params=[self.W, self.B], lr = param_delta)
        for i in range(param_niter + 1):
            loss = self.get_loss(X, Yoh_, param_lambdba)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if i % 10 == 0:
                print(f'step: {i}\tloss: {loss:.6f}')

    def eval(model, X: np.array) -> np.array:
        """Arguments:
            - model: type: PTLogreg
            - X: actual datapoints [NxD], type: np.array
            Returns: predicted class probabilites [NxC], type: np.array
        """
        probs = torch.tensor(X)
        argmaxes = torch.argmax(probs, dim=1)
        return np.array(argmaxes)

if __name__ == "__main__":
    np.random.seed(100)
    
    C = 2
    X, Y_ = sample_gmm_2d(2, C, 10)
    Yoh_ = class_to_onehot(Y_)

    ptlr = PTLogreg(X.shape[1], Yoh_.shape[1])
    ptlr.train(torch.tensor(X), torch.tensor(Yoh_), 10000, 0.05, 1e-3)
    Y = ptlr.eval(X)

    accuracy, recall, precision = eval_perf_multi(Y, Y_)
    print(f'accuracy: {accuracy:.4f}')
    for i in range(C):
        print(f'class #{i} recall: {recall[i]}\tprecision: {precision[i]}')

    box = (np.min(X, axis=0) - 0.5, np.max(X, axis=0) + 0.5)
    graph_surface(ptlr.eval, box, 0.5, 1024, 1024)
    graph_data(torch.tensor(X), torch.tensor(Y_), torch.tensor(Y))
    plt.show()