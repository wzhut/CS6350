import tensorflow as tf
import numpy as np

class TFNN():
    def __init__(self, width):
        self.width = width
        
        self.n_layers = len(width)
        self.in_d = width[0]
        self.out_d = width[-1]
        self.epoch = 10000

        self.X = tf.placeholder(tf.float32, [None, self.in_d])
        self.Y_train = tf.placeholder(tf.float32, [None, self.out_d])

        # weights
        # zero initialization
        # self.w = [tf.Variable(tf.zeros([self.width[i-1], self.width[i] - 1])) for i in range(1, self.n_layers-1)]
        # i = self.n_layers - 1
        # self.w.append(tf.Variable(tf.zeros([self.width[i-1], self.width[i]])))
        # Xavier initialization
        self.w = [tf.Variable(tf.random_normal([self.width[i-1], self.width[i] - 1], stddev = np.sqrt(1 / (self.width[i-1] + self.width[i] - 1)))) for i in range(1, self.n_layers-1)]
        i = self.n_layers - 1
        self.w.append(tf.Variable(tf.random_normal([self.width[i-1], self.width[i]], stddev = np.sqrt(1 / (self.width[i-1] + self.width[i]) ))))
        # HE initialization
        # self.w = [tf.Variable(tf.random_normal([self.width[i-1], self.width[i] - 1], stddev = np.sqrt(2 / (self.width[i-1])))) for i in range(1, self.n_layers-1)]
        # i = self.n_layers - 1
        # self.w.append(tf.Variable(tf.random_normal([self.width[i-1], self.width[i]], stddev = np.sqrt(2 / (self.width[i-1]) ))))

        # forward
        n_sample = tf.shape(self.X)[0]
        z = self.X
        for i in range(1, self.n_layers):            
            a = tf.matmul(z, self.w[i - 1])
            # z = tf.nn.relu(a)
            z = tf.nn.tanh(a)
            z = tf.concat([z, tf.ones([n_sample, 1])], 1)
            if i == self.n_layers - 1:
                self.Y = a

        self.Loss = tf.reduce_sum((self.Y - self.Y_train)**2)
        self.opt = tf.train.AdamOptimizer().minimize(self.Loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
    def train(self, train_x, train_y):
        feed_dict = {self.X: train_x, self.Y_train: train_y}
        for i in range(self.epoch):
            _,loss =  self.sess.run([self.opt, self.Loss], feed_dict=feed_dict)
            # if i % 100 == 0:
            #     print('iter: ', i, 'Loss: ', loss)

    def fit(self, test_x):
        feed_dict = {self.X: test_x}
        y_pred = self.sess.run(self.Y, feed_dict=feed_dict)
        # print(y_pred)
        return y_pred