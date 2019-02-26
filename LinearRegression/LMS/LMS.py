import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LMS:
    # method: 0 batch gradient descent
    #         1 SGD
    #         2 normal equation
    def __init__(self):
        self.method = 0
        self.lr = 0.01
        self.threshold = 1e-5
        self.max_iter = 1000

    def set_method(self, method):
        self.method = method

    def set_lr(self, lr):
        self.lr = lr
    
    def set_threshold(self, threshold):
        self.threshold = threshold
    
    def set_max_iter(self, max_iter):
        self.max_iter = max_iter
    
    
    def optimize(self, x, y):
        if self.method == 0:
            return self.optimize_GD(x,y)
        elif self.method == 1:
            return self.optimize_SGD(x,y)
        elif self.method == 2:
            return self.optimize_NE(x,y)

    # x is augmented
    def optimize_GD(self, x, y):
        dim = x.shape[1]
        # update difference
        diff = 1
        # init w
        w = np.zeros([dim,1])
        fun_val = []
        it = 0
        while diff > self.threshold and it < self.max_iter:
            it = it + 1
            tmp = np.reshape(np.squeeze(np.matmul(x,w)) - y, (-1,1))
            g = np.reshape(np.sum(np.transpose(tmp*x), axis=1), (-1,1))
            delta = -self.lr * g
            w_new = w + delta
            diff = np.sqrt(np.sum(np.square(delta)))
            w = w_new
            tmp = np.reshape(np.squeeze(np.matmul(x,w)) - y, (-1,1))
            fv = 0.5 * np.sum(np.square(tmp))
            fun_val.append(fv)

        # save func_val iter plot
        fig = plt.figure()
        fig.suptitle('Gradient Descent')
        plt.xlabel('Iteration', fontsize=18)
        plt.ylabel('Cost Function Value', fontsize=16)
        plt.plot(fun_val, 'b') 
        plt.legend(['train'])
        fig.savefig('GD.png')

        return w
    
    def optimize_SGD(self, x, y):
        dim = x.shape[1]
        n = x.shape[0]
        # update difference
        diff = 1
        # init w
        w = np.zeros([dim,1])
        fv = 1
        fun_val = []
        it = 0
        while fv > self.threshold:
            it = it + 1
            idx = np.random.randint(n,size=1)
            x1 = x[idx]
            y1 = y[idx]
            g = np.sum(np.transpose((np.matmul(x1,w) - y1)*x1), axis=1)
            delta = -self.lr * np.reshape(g,(-1,1))
            w_new = w + delta
            diff = np.sqrt(np.sum(np.square(delta)))
            w = w_new
            tmp = np.reshape(np.squeeze(np.matmul(x,w)) - y, (-1,1))
            fv = 0.5 * np.sum(np.square(tmp))
            fun_val.append(fv)
        
        # save func_val iter plot
        fig = plt.figure()
        fig.suptitle('Stochastic Gradient Descent')
        plt.xlabel('Iteration', fontsize=18)
        plt.ylabel('Cost Function Value', fontsize=16)
        plt.plot(fun_val, 'b') 
        plt.legend(['train'])
        fig.savefig('SGD.png')

        return w

    def optimize_NE(self, x, y):
        x_t = np.transpose(x)
        t1 = np.linalg.inv(np.matmul(x_t, x))
        t2 = np.matmul(x_t, y)
        return np.matmul(t1,t2)
        