import torch
import torch.optim as optim

class LinearRegression:
    def __debug_grads(self, grad_a_troch, grad_a_ours, grad_b_torch, grad_b_ours):
        print(f'a\t pytorch:{grad_a_troch:.6f}\t ours:{grad_a_ours:.6f}')
        print(f'b\t pytorch:{grad_b_torch:.6f}\t ours:{grad_b_ours:.6f}')
    
    def train(self, X=[1, 2], Y_=[3, 5], ntimes = 100, delta = 0.1):
        self.X = torch.Tensor(X)
        self.N = len(X)
        self.Y_ = torch.tensor(Y_)
        self.ntimes = ntimes
        self.delta = delta

        self.a = torch.randn(1, requires_grad=True)
        self.b = torch.randn(1, requires_grad=True)
        optimizer = optim.SGD([self.a, self.b], lr = self.delta)
        for i in range(1, self.ntimes + 1):
            pred_y = self.a * self.X + self.b
            diff = (self.Y_ - pred_y)
            loss = torch.mean(diff ** 2)

            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                dL_da = (2 / self.N) * torch.sum((pred_y - self.Y_) * self.X) 
                dL_db = (2 / self.N) * torch.sum(pred_y - self.Y_)
                print(f'step: {i}\tloss: {loss:.6f}\tpredicted_y: {pred_y.detach()}\ta: {self.a.detach()[0]:.4f}\tb: {self.b.detach()[0]:.4f}')
                self.__debug_grads(self.a.grad.detach()[0], dL_da, self.b.grad.detach()[0], dL_db)
            
            optimizer.zero_grad()

        return Y_

    def classify(self, X):
        X = torch.tensor(X)
        return self.a * X + self.b

# Zadatak 3
if __name__ == '__main__':
    lr = LinearRegression()
    func = lambda x : 3 * x + 4
    __N = 20

    X = [x for x in range(0, __N)]
    Y_= [func(x) for x in X]
    _ = lr.train(X, Y_, 10000, 0.005)