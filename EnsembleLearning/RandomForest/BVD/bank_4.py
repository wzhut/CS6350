#%%
import pandas as pd
import DT as dt
import numpy as np
import matplotlib.pyplot as plt

# load train data
columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
types = {'age': int, 
        'job': str, 
        'marital': str, 
        'education': str,
        'default': str,
        'balance': int,
        'housing': str,
        'loan': str,
        'contact': str,
        'day': int,
        'month': str,
        'duration': int,
        'campaign': int,
        'pdays': int,
        'previous': int,
        'poutcome': str,
        'y': str}
# load train data
train_data =  pd.read_csv('../../data/bank/train.csv', names=columns, dtype=types)
train_size = len(train_data.index)
## process data
# convert numeric to binary
numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
for c in numeric_features:
    median = train_data[c].median()
    train_data[c] = train_data[c].apply(lambda x: 0 if x < median else 1)

# replace unknowns
# unknown_features = ['job', 'education', 'contact', 'poutcome']
# for c in unknown_features:
#     order = train_data[c].value_counts().index.tolist()
#     if order[0] != 'unknown':
#         replace = order[0]
#     else:
#         replace = order[1]
#     train_data[c] = train_data[c].apply(lambda x: replace if x == 'unknown' else x)
#load test data
test_data =  pd.read_csv('../../data/bank/test.csv', names=columns, dtype=types)
test_size = len(test_data.index)
for c in numeric_features:
    median = test_data[c].median()
    test_data[c] = test_data[c].apply(lambda x: 0 if x < median else 1)

# for c in unknown_features:
#     order = test_data[c].value_counts().index.tolist()
#     if order[0] != 'unknown':
#         replace = order[0]
#     else:
#         replace = order[1]
#     test_data[c] = test_data[c].apply(lambda x: replace if x == 'unknown' else x)

# set features and label
features = {'age': [0, 1],  # converted to binary
        'job': ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student', 'blue-collar', 'self-employed', 'retired', 'technician', 'services'], 
        'marital': ['married','divorced','single'], 
        'education': ['unknown', 'secondary', 'primary', 'tertiary'],
        'default': ['yes', 'no'],
        'balance': [0, 1],  # converted to binary
        'housing': ['yes', 'no'],
        'loan': ['yes', 'no'],
        'contact': ['unknown', 'telephone', 'cellular'],
        'day': [0, 1],  # converted to binary,
        'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
        'duration': [0, 1],  # converted to binary
        'campaign': [0, 1],  # converted to binary
        'pdays': [0, 1],  # converted to binary
        'previous': [0, 1],  # converted to binary
        'poutcome': ['unknown', 'other', 'failure', 'success']}
label = {'y': ['yes', 'no']}

num_run = 100
T = 1000

test_py = np.array([[0 for x in range(test_size)] for y in range(num_run)])
test_py_first = np.array([0 for x in range(test_size)])

for iter in range(num_run):
        train_subset = train_data.sample(n=1000, replace=False, random_state=iter)
        for t in range(T):
                print('iter: ', iter, 't: ',t)
                # sample with replace
                sampled = train_subset.sample(frac=0.01, replace=True, random_state=t)
                # ID3
                dt_generator = dt.ID3(feature_selection=0, max_depth=17, subset=4)
                # get decision tree
                decision_tree = dt_generator.generate_decision_tree(sampled, features, label)
                ## predict
                # test
                py = dt_generator.classify(decision_tree, test_data) 
                py = np.array(py.tolist())
                py[py == 'yes'] = 1
                py[py == 'no'] = -1
                py = py.astype(int)
                test_py[iter] = test_py[iter] + py
                if t == 0:
                        test_py_first = test_py_first + py

true_value = np.array(test_data['y'].tolist())
true_value[true_value == 'yes'] = 1
true_value[true_value == 'no'] = -1
true_value = true_value.astype(int)

# frist tree predictor
# take average
test_py_first = test_py_first / num_run
# bias
bias = np.mean(np.square(test_py_first - true_value))
# variance
mean = np.mean(test_py_first) 
variance = np.sum(np.square(test_py_first - mean)) / (test_size - 1)
se = bias + variance
print(bias)
print(variance)
print('100 single tree predictor: ', se)
# bagged tree predictor
# take average
test_py = np.sum(test_py,axis=0) / (num_run * T)
# bias
bias = np.mean(np.square(test_py - true_value))
# variance
mean = np.mean(test_py)
variance = np.sum(np.square(test_py - mean)) / (test_size - 1)
se = bias + variance
print(bias)
print(variance)
print('100 Random Forest predictor:', se)




