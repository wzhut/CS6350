import numpy as np
## sigmoid neural network
class NN:
    def __init__(self, width):
        self.in_d = width[0]
        self.out_d = width[-1]
        self.lr = 0.1
        self.d = 0.1
        self.epoch = 100
        self.gamma = 0.1

        # width including input and output
        self.width = width
        self.layers = len(width)
        # weights
        self.w = [None for _ in range(self.layers)]
        self.dw = [None for _ in range(self.layers)]
        for i in range(1, self.layers-1):
            wi = np.random.normal(0, 1, (self.width[i] - 1, self.width[i-1]))
            # wi = np.zeros([self.width[i] - 1, self.width[i-1]])
            self.w[i] = wi
            self.dw[i] = np.zeros([self.width[i] - 1, self.width[i - 1]])
        i = self.layers - 1
        wi = np.random.normal(0, 1, (self.width[i], self.width[i-1]))
        self.w[i] = wi
        self.dw[i] = np.zeros([self.width[i], self.width[i - 1]])
        # nodes
        self.nodes = [np.ones([self.width[i], 1]) for i in range(self.layers)]
        # self.dnodes = [np.zeros([self.width[i], 1]) for i in range(self.layers)]
        #

    def train(self, x, y):
        num_sample = x.shape[0]
        dim = x.shape[1]
        idx = np.arange(num_sample)
        for t in range(self.epoch):
            np.random.shuffle(idx)
            x = x[idx,:]
            y = y[idx]
            for i in range(num_sample):
                self.forward_backward(x[i,:].reshape([self.in_d, 1]), y[i,:].reshape([self.out_d, 1]))
                lr = self.gamma / (1 + self.gamma/self.d * t)
                self.update_w(lr)

    def update_w(self, lr):
        for i in range(1, self.layers):
            self.w[i] = self.w[i] - self.lr * self.dw[i]
        
    def forward(self, x):
        # input
        self.nodes[0] = x
        for i in range(1, self.layers-1):
            self.nodes[i][:-1,:] = self.sigmoid(np.matmul(self.w[i], self.nodes[i - 1]).reshape([-1,1]))
        # output
        i = self.layers - 1
        self.nodes[i] = np.matmul(self.w[i], self.nodes[i - 1])
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, y):
        # output
        # dLdy
        dLdz = self.nodes[-1] - y
        nk = self.width[-1]
        dzdw =  np.transpose(np.tile(self.nodes[-2], [1, nk]))
        self.dw[-1] = dLdz * dzdw
        # dydz
        dzdz = self.w[-1][:, :-1]
        # derivative between k-1 and k
        for i in reversed(range(1, self.layers - 1)):
            # sigmod derivative z = sigmoid(a)
            nk = self.width[i] - 1
            # ignore bias term in k
            z_in = self.nodes[i-1]
            z_out = self.nodes[i][:-1]
            dadw = np.transpose(np.tile(z_in, [1, nk]))
            dzdw = z_out * (1 - z_out) * dadw
            dLdz = np.matmul(np.transpose(dzdz), dLdz)
            dLdw = dLdz * dzdw
            self.dw[i] = dLdw

            dzdz = z_out * (1 - z_out) * self.w[i] 
            dzdz = dzdz[:, :-1]

    def forward_backward(self, x, y):
        self.forward(x)
        self.backward(y)
    
    def fit(self, x):
        num_sample = x.shape[0]
        l = []
        for i in range(num_sample):
            self.forward(x[i,:].reshape(self.in_d))
            y = self.nodes[-1]
            l.append(np.transpose(y))
        y_pred = np.concatenate(l, axis=0)
        return y_pred