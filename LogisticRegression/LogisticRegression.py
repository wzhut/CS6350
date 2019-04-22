import numpy as np

class LogisticRegression:
    def __init__(self):
        self.lr = 0.01
        self.d = 0.1
        self.epoch = 100
        self.gamma = 0.1
        self.v = 1
    
    def set_lr(self, lr):
        self.lr = lr
    
    def set_d(self, d):
        self.d = d
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def set_gamma(self, gamma):
        self.gamma = gamma
    
    def set_v(self, v):
        self.v = v
    
    def train_MAP(self, x, y):
        num_sample = x.shape[0]
        dim = x.shape[1]
        w = np.zeros([1, dim])
        idx = np.arange(num_sample)
        for t in range(self.epoch):
            np.random.shuffle(idx)
            x = x[idx,:]
            y = y[idx]
            for i in range(num_sample):
                x_i = x[i,:].reshape([1, -1])
                tmp = y[i] * np.sum(np.multiply(w, x_i))
                g = - num_sample * y[i] * x_i / (1 + np.exp(tmp)) + w / self.v
                # print(g)
                lr = self.lr / (1 + self.lr / self.d * t)
                w = w - lr * g
        return w.reshape([-1,1])
    
    def train_ML(self, x, y):
        num_sample = x.shape[0]
        dim = x.shape[1]
        w = np.zeros([1, dim])
        idx = np.arange(num_sample)
        for t in range(self.epoch):
            np.random.shuffle(idx)
            x = x[idx,:]
            y = y[idx]
            for i in range(num_sample):
                tmp = y[i] * np.sum(np.multiply(w, x[i,:]))
                g = - num_sample * y[i] * x[i,:] / (1 + np.exp(tmp))
                lr = self.lr / (1 + self.lr / self.d * t)
                w = w - lr * g
        return w.reshape([-1,1])