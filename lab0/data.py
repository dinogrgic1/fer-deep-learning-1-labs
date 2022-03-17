import numpy as np
import matplotlib.pyplot as plt

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
    # create the distributions and groundtruth labels
    distribution = []
    ys = []
    for i in range(C):
        distribution.append(Random2DGaussian())
        ys.append(i)

    # sample the dataset
    x_matrix = np.vstack([G.get_sample(N) for G in distribution])
    y_matrix = np.hstack([[Y]*N for Y in ys])
    return x_matrix, y_matrix

def eval_perf_binary(Y,Y_):
    N = len(Y_)
    TP = ((Y == 1) & (Y_ == 1)).sum()
    FP = ((Y == 1) & (Y_ == 0)).sum()
    FN = ((Y == 0) & (Y_ == 1)).sum()

    accuracy = (Y_ == Y).sum() / N
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    
    return accuracy, recall, precision

def eval_AP(Y):
    N = len(Y)
    TP = sum(Y)
    FP = N - sum(Y)
    
    sumprec = 0
    for x in Y:
        precision = TP / (TP + FP)
        if x:
            sumprec += precision
        TP -= x
        FP -= not x
    return sumprec / sum(Y)

def graph_surface(function, rect, offset=0.5, width=256, height=256):

  lsw = np.linspace(rect[0][1], rect[1][1], width) 
  lsh = np.linspace(rect[0][0], rect[1][0], height)
  xx0,xx1 = np.meshgrid(lsh, lsw)
  grid = np.stack((xx0.flatten(),xx1.flatten()), axis=1)

  values=function(grid).reshape((width,height))
  
  delta = offset if offset else 0
  maxval=max(np.max(values)-delta, - (np.min(values)-delta))
  
  plt.pcolormesh(xx0, xx1, values, 
     vmin=delta-maxval, vmax=delta+maxval)
    
  if offset != None:
    plt.contour(xx0, xx1, values, colors='black', levels=[offset])

# TODO(Dino): Some bugs need to be fixed here
def graph_data(X, Y_, Y, special=[]):
    palette=([0.5, 0.5, 0.5], [1, 1, 1], [0.2, 0.2, 0.2])
    colors = np.tile([1.0,1.0,.0], (Y_.shape[0], 1))
    for i in range(len(palette)):
        colors[Y_==i] = palette[i]

    sizes = np.repeat(20, len(Y_))
    sizes[special] = 40
    
    good = (Y_ == Y)
    plt.scatter(X[good,0] ,X[good,1], c='grey', s=sizes[good], marker='s', edgecolors='black')

    bad = (Y_ != Y)
    plt.scatter(X[bad,0], X[bad,1], c='white', s=sizes[bad], marker='o', edgecolors='black')

def myDummyDecision(X):
    scores = X[:,0] + X[:,1] - 5
    return scores

if __name__=="__main__":
    np.random.seed(100)
  
    X, Y_ = sample_gauss_2d(2, 100)
    Y = myDummyDecision(X) > 0.5

    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(myDummyDecision, bbox, offset=0)
    
    graph_data(X, Y_, Y) 
  
    plt.show()