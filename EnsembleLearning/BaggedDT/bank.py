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
train_data =  pd.read_csv('../data/bank/train.csv', names=columns, dtype=types)
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
test_data =  pd.read_csv('../data/bank/test.csv', names=columns, dtype=types)
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

T = 1000

train_err = [0 for x in range(T)]
test_err = [0 for x in range(T)]
train_py = np.array([0 for x in range(train_size)])
test_py = np.array([0 for x in range(test_size)])

for t in range(T):
    print()
    # sample with replace
    sampled = train_data.sample(frac=0.5, replace=True, random_state=t)
    # ID3
    dt_generator = dt.ID3(feature_selection=0, max_depth=17)
    # get decision tree
    decision_tree = dt_generator.generate_decision_tree(sampled, features, label)

    ## predict
    # train
    py = dt_generator.classify(decision_tree, train_data) 
    py = np.array(py.tolist())
    py[py == 'yes'] = 1
    py[py == 'no'] = -1
    py = py.astype(int)
    train_py = train_py + py
    py = py.astype(str)
    py[train_py > 0] = 'yes'
    py[train_py <=0] = 'no'
    train_data['py'] = pd.Series(py)

    acc = train_data.apply(lambda row: 1 if row['y'] == row['py'] else 0, axis=1).sum() / train_size
    err = 1 - acc
    train_err[t] = err
    # test
    py = dt_generator.classify(decision_tree, test_data) 
    py = np.array(py.tolist())
    py[py == 'yes'] = 1
    py[py == 'no'] = -1
    py = py.astype(int)
    test_py = test_py + py
    py = py.astype(str)
    py[test_py > 0] = 'yes'
    py[test_py <=0] = 'no'
    test_data['py'] = pd.Series(py)
    acc = test_data.apply(lambda row: 1 if row['y'] == row['py'] else 0, axis=1).sum() / test_size
    err = 1 - acc
    test_err[t] = err
    print('t: ', t, 'train_err: ', train_err[t], 'test_err: ', test_err[t])


fig = plt.figure()
fig.suptitle('Bagged Decision Tree')
plt.xlabel('Iteration', fontsize=18)
plt.ylabel('Error Rate', fontsize=16)
plt.plot(train_err, 'b')
plt.plot(test_err, 'r')  
plt.legend(['train', 'test'])
fig.savefig('bdt.png')




# print('train_acc:', train_acc)
# print('test_acc:', test_acc)



