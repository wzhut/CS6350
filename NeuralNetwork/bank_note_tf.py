#%%
import pandas as pd
import numpy as np
import TFNN

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

# train_x = np.array([[0.5, -1, -3, 1], [-1,-2,-2,1], [1.5, 0.2, -2.5, 1]])
# train_y = np.array([1,-1,1])

gamma_set = np.array([0.01, 0.1, 0.5, 1, 2, 5, 10, 100])
in_d = train_x.shape[1]
out_d = 1

width_list = [5, 10, 25, 50, 100]
n_layers = [3, 5, 9]
for n_layer in n_layers:
    for width in width_list:
        print('layer: ', n_layer, 'width: ', width)
        s = [in_d, out_d]
        for _ in range(n_layer):
            s.insert(1, width)

        model= TFNN.TFNN(s)

        model.train(train_x.reshape([-1, in_d]), train_y.reshape([-1,1]))
        pred = model.fit(train_x)

        pred[pred > 0] = 1
        pred[pred <= 0] = -1
        train_err = np.sum(np.abs(pred - np.reshape(train_y,(-1,1)))) / 2 / train_y.shape[0]

        pred = model.fit(test_x)
        pred[pred > 0] = 1
        pred[pred <= 0] = -1

        test_err = np.sum(np.abs(pred - np.reshape(test_y,(-1,1)))) / 2 / test_y.shape[0]
        print('train_error: ', train_err, ' test_error: ', test_err)