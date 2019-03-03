import pandas as pd
import numpy as np

class Perceptron:
    def __init__(self):
        self.lr = 0.1
        self.T = 10

    def std_alg(self, x, y):
        num_sample = x.shape[0]
        dim = x.shape[1]
        w = np.zeros(dim)
        idx = np.arange(num_sample)
        for t in range(self.T):
            np.random.shuffle(idx)
            x = x[idx,:]
            y = y[idx]
            for i in range(num_sample):
                tmp = np.sum(x[i] * w)
                if not (tmp * y[i] > 0):
                    w = w + self.lr * y[i] * x[i]
        return w

    def voted_alg(self, x, y):
        num_sample = x.shape[0]
        dim = x.shape[1]
        w = np.zeros(dim)
        idx = np.arange(num_sample)
        c_list = np.array([])
        w_list = np.array([])
        c = 0
        for t in range(self.T):
            np.random.shuffle(idx)
            x = x[idx,:]
            y = y[idx]
            for i in range(num_sample):
                tmp = np.sum(x[i] * w)
                if not (tmp * y[i] > 0):
                    w_list = np.append(w_list, w)
                    c_list = np.append(c_list, c)
                    w = w + self.lr * y[i] * x[i]
                    c = 1
                else:
                    c = c + 1
        num = c_list.shape[0]
        w_list = np.reshape(w_list, (num,-1))
        return c_list, w_list
    
    def avg_alg(self, x, y):
        num_sample = x.shape[0]
        dim = x.shape[1]
        w = np.zeros(dim)
        a = np.zeros(dim)
        idx = np.arange(num_sample)
        for t in range(self.T):
            np.random.shuffle(idx)
            x = x[idx,:]
            y = y[idx]
            for i in range(num_sample):
                tmp = np.sum(x[i] * w)
                if not (tmp * y[i] > 0):
                    w = w + self.lr * y[i] * x[i]
                a = a + w
        return a
        