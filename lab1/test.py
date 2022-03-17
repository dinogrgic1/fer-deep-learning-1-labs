from data import *
from fcann2 import *

__K = 6
__C = 2
__N = 10
__param_niter = 1e5
__param_delta = 0.05
__param_lambda = 1e-3
__nhidden = 5

if __name__ == '__main__':
    np.random.seed(100)

    X, Y_ = sample_gmm_2d(__K, __C, __N)
    fcann2_train(X, Y_, __param_niter, __param_delta, __param_lambda, __nhidden)