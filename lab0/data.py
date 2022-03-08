import numpy as np

class Random2DGaussian:
    minx=0
    maxx=10
    miny=0
    maxy=10
    scalecov=5
    mean = None
    covv = None

    def __init__(self, minx = 0, maxx = 10, miny = 0, maxy = 10) -> None:
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy

        self.mean = np.random.random_sample(2) * (self.maxx - self.minx, self.maxy - self.miny)
        self.mean += (self.minx, self.miny)
        
        eigvals = np.random.random_sample(2)
        eigvals *= ((self.maxx - self.minx) / self.scalecov, (self.maxy - self.miny) / self.scalecov)
        eigvals **= 2
        D = np.diag(eigvals)

        angle = np.random.random_sample() * np.pi* 2
        R = np.matrix([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        self.covv = np.transpose(R) * D * R

    def get_sample(self, n) -> np.ndarray:
        return np.random.multivariate_normal(self.mean, self.covv, n)

def sample_gauss_2d(C : int, N : int):
    x_matrix = np.ndarray([C * N, 2]) 
    y_matrix = np.ndarray([C * N, 1])

    for i in range(0, C):
        r = Random2DGaussian()
        x_matrix[i * N: (i + 1) * N] = r.get_sample(N)
        y_matrix[i * N: (i + 1) * N] = np.full([N, 1], fill_value=i)
    return x_matrix, y_matrix
