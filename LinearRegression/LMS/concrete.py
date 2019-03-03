#%%
import pandas as pd
import numpy as np
import LMS

train_data = pd.read_csv('../data/concrete/train.csv', header=None)
# process data
raw = train_data.values
num_col = raw.shape[1]
num_row = raw.shape[0]
train_x = np.copy(raw)
train_x[:,num_col - 1] = 1
train_y = raw[:, num_col - 1]
train_y = 2 * train_y - 1

test_data = pd.read_csv('../data/concrete/test.csv', header=None)
raw = test_data.values
num_col = raw.shape[1]
num_row = raw.shape[0]
test_x = np.copy(raw)
test_x[:,num_col - 1] = 1
test_y = raw[:, num_col - 1]
test_y = 2 * test_y - 1

# lms model
lms = LMS.LMS()

# GD
w = lms.optimize(train_x, train_y)
print('GD w: ', w)
tmp = np.reshape(np.squeeze(np.matmul(test_x,w)) - test_y, (-1,1))
fv = 0.5 * np.sum(np.square(tmp))
print('GD test_fv: ', fv)
# SGD
lms.set_method(1)
w = lms.optimize(train_x, train_y)
print('SGD w: ', w)
tmp = np.reshape(np.squeeze(np.matmul(test_x,w)) - test_y, (-1,1))
fv = 0.5 * np.sum(np.square(tmp))
print('GD test_fv: ', fv)
# normal equation
lms.set_method(2)
w = lms.optimize(train_x, train_y)
print('NE w: ', w)
tmp = np.reshape(np.squeeze(np.matmul(test_x,w)) - test_y, (-1,1))
fv = 0.5 * np.sum(np.square(tmp))
print('NE test_fv: ', fv)
