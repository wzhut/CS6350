#%%
import pandas as pd
import numpy as np
import SVM

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

C_set = np.array([100, 500, 700])
C_set = C_set / 873
gamma_set = np.array([0.01, 0.1, 0.5, 1, 2, 5, 10, 100])
svm = SVM.SVM()
for C in C_set:
    print('C: ', C)
    svm.set_C(C)
    w = svm.train_p(train_x, train_y)
    w = np.reshape(w, (5,1))

    pred = np.matmul(train_x, w)
    pred[pred > 0] = 1
    pred[pred <= 0] = -1
    train_err = np.sum(np.abs(pred - np.reshape(train_y,(-1,1)))) / 2 / train_y.shape[0]

    pred = np.matmul(test_x, w)
    pred[pred > 0] = 1
    pred[pred <= 0] = -1

    test_err = np.sum(np.abs(pred - np.reshape(test_y,(-1,1)))) / 2 / test_y.shape[0]
    print('linear SVM Primal train_error: ', train_err, ' test_error: ', test_err)
    w = np.reshape(w, (1,-1))
    # print('w1: ', w)

    # dual form
    w = svm.train_d(train_x[:,[x for x in range(num_col - 1)]] ,train_y)
    # print('w2: ', w)

    w = np.reshape(w, (5,1))

    pred = np.matmul(train_x, w)
    pred[pred > 0] = 1
    pred[pred <= 0] = -1
    train_err = np.sum(np.abs(pred - np.reshape(train_y,(-1,1)))) / 2 / train_y.shape[0]

    pred = np.matmul(test_x, w)
    pred[pred > 0] = 1
    pred[pred <= 0] = -1

    test_err = np.sum(np.abs(pred - np.reshape(test_y,(-1,1)))) / 2 / test_y.shape[0]
    print('linear SVM Dual train_error: ', train_err, ' test_error: ', test_err)

    # gaussian kernel
    c = 0
    for gamma in gamma_set:
        print('gamma: ', gamma)
        svm.set_gamma(gamma)
        alpha = svm.train_gaussian_kernel(train_x[:,[x for x in range(num_col - 1)]] ,train_y)
        idx = np.where(alpha > 0)[0]
        print('#sv: ', len(idx))
        # train 
        y = svm.predict_gaussian_kernel(alpha, train_x[:,[x for x in range(num_col - 1)]], train_y, train_x[:,[x for x in range(num_col - 1)]])
        train_err = np.sum(np.abs(y - np.reshape(train_y,(-1,1)))) / 2 / train_y.shape[0]

        # test
        y = svm.predict_gaussian_kernel(alpha, train_x[:,[x for x in range(num_col - 1)]], train_y, test_x[:,[x for x in range(num_col - 1)]])
        test_err = np.sum(np.abs(y - np.reshape(test_y,(-1,1)))) / 2 / test_y.shape[0]
        print('nonlinear SVM train_error: ', train_err, ' test_error: ', test_err)
        
        if c > 0:
            intersect = len(np.intersect1d(idx, old_idx))
            print('#intersect: ', intersect)
        c = c + 1
        old_idx = idx