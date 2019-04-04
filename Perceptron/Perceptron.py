import pandas as pd
import numpy as np

class Perceptron:
    def __init__(self):
        self.lr = 0.1
        self.T = 10
        self.gamma = 0.1

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
    
    def gaussian_kernel(self, x1, x2, gamma):
        m1 = np.tile(x1, (1, x2.shape[0]))
        m1 = np.reshape(m1, (-1,x1.shape[1]))
        m2 = np.tile(x2, (x1.shape[0], 1))
        k = np.exp(np.sum(np.square(m1 - m2),axis=1) / -gamma)
        k = np.reshape(k, (x1.shape[0], x2.shape[0]))
        return k
    
    def set_gamma(self, gamma):
        self.gamma = gamma
    
    def kernel(self, x, y):
        num_sample = x.shape[0]
        idx = np.arange(num_sample)
        c = np.array([x for x in range(num_sample)])
        c = np.reshape(c, (-1, 1))
        y = np.reshape(y, (-1, 1))
        k = self.gaussian_kernel(x,x, self.gamma)
        for t in range(self.T):
            np.random.shuffle(idx)
            for i in range(num_sample):
                cy = np.multiply(c, y)
                cyk = np.matmul(k[idx[i], :], cy)
                if cyk * y[idx[i]] <= 0:
                    c[idx[i]] = c[idx[i]] + 1
        return c
    
    def kernel_predict(self, c, x0, y0, x):
        k = self.gaussian_kernel(x0, x, self.gamma)
        cy = np.reshape(np.multiply(c, np.reshape(y0, (-1, 1))), (1, -1))
        y = np.matmul(cy, k)
        y = np.reshape(y, (-1,1))
        y[y > 0] = 1
        y[y <=0] = -1
        return y
        