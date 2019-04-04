#%%
import pandas as pd
import numpy as np
import Perceptron

train_data = pd.read_csv('../data/bank-note/train.csv', header=None)
# process data
raw = train_data.values
num_col = raw.shape[1]
num_row = raw.shape[0]
train_x = np.copy(raw)
train_x[:,num_col - 1] = 1
train_y = raw[:, num_col - 1]
train_y = 2 * train_y - 1

test_data = pd.read_csv('../data/bank-note/test.csv', header=None)
raw = test_data.values
num_col = raw.shape[1]
num_row = raw.shape[0]
test_x = np.copy(raw)
test_x[:,num_col - 1] = 1
test_y = raw[:, num_col - 1]
test_y = 2 * test_y - 1

p = Perceptron.Perceptron()
#standard algorithm
w = p.std_alg(train_x, train_y)
w = np.reshape(w, (-1,1))
pred = np.matmul(test_x, w)
pred[pred > 0] = 1
pred[pred <= 0] = -1
err = np.sum(np.abs(pred - np.reshape(test_y,(-1,1)))) / 2 / test_y.shape[0]
print('standard: ', err)
print(w)
# voting
c_list, w_list = p.voted_alg(train_x, train_y)
c_list = np.reshape(c_list, (-1,1))
print(w_list)
w_list = np.transpose(w_list)
prod = np.matmul(test_x, w_list)
prod[prod >0] = 1
prod[prod <=0] = -1
voted = np.matmul(prod, c_list)
voted[voted >0] = 1
voted[voted<=0] = -1
err = np.sum(np.abs(voted - np.reshape(test_y,(-1,1)))) / 2 / test_y.shape[0]
print('voted: ', err)
print(c_list)

# average
w = p.avg_alg(train_x, train_y)
w = np.reshape(w, (-1,1))
pred = np.matmul(test_x, w)
pred[pred > 0] = 1
pred[pred <= 0] = -1
err = np.sum(np.abs(pred - np.reshape(test_y,(-1,1)))) / 2 / test_y.shape[0]
print('averaged: ', err)
print(w)

# kernel
gamma_set = np.array([0.01, 0.1, 0.5, 1, 2, 5, 10, 100])
for gamma in gamma_set:
    print('gamma: ', gamma)
    p.set_gamma(gamma)
    c = p.kernel(train_x, train_y)
    # train 
    y = p.kernel_predict(c, train_x, train_y, train_x)
    train_err = np.sum(np.abs(y - np.reshape(train_y,(-1,1)))) / 2 / train_y.shape[0]

    # test
    y = p.kernel_predict(c, train_x, train_y, test_x)
    test_err = np.sum(np.abs(y - np.reshape(test_y,(-1,1)))) / 2 / test_y.shape[0]
    print('nonlinear SVM train_error: ', train_err, ' test_error: ', test_err)   
