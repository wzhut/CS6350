#%%
import pandas as pd
import DT as dt

# load train data
columns = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15','X16','X17','X18','X19','X20','X21','X22','X23','Y']
types = {'X1': int, 
        'X2': int,
        'X3': int,
        'X4': int,
        'X6': int,
        'X7': int,
        'X8': int,
        'X9': int,
        'X10': int,
        'X11': int,
        'X12': int,
        'X13': int,
        'X14': int,
        'X15': int,
        'X16': int,
        'X17': int,
        'X18': int,
        'X19': int,
        'X20': int,
        'X21': int,
        'X22': int,
        'X23': int,
        'Y': int}
# load train data
train_data =  pd.read_csv('../data/credit/train.csv', names=columns, dtype=types)
train_size = len(train_data.index)
## process data
# convert numeric to binary
numeric_features = ['X1', 'X5', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23']
for c in numeric_features:
    median = train_data[c].median()
    train_data[c] = train_data[c].apply(lambda x: 0 if x < median else 1)

test_data =  pd.read_csv('../data/credit/test.csv', names=columns, dtype=types)
test_size = len(test_data.index)
for c in numeric_features:
    median = test_data[c].median()
    test_data[c] = test_data[c].apply(lambda x: 0 if x < median else 1)


# set features and label
features = {'X1': [0, 1],  # converted to binary
        'X2': [1, 2], 
        'X3': [0,1,2,3,4,5,6], 
        'X4': [0,1,2,3],
        'X5': [0, 1],
        'X6': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
        'X7': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
        'X8': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
        'X9': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
        'X10': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
        'X11': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
        'X12': [0, 1],
        'X13': [0, 1],
        'X14': [0, 1],
        'X15': [0, 1],
        'X16': [0, 1],
        'X17': [0, 1],
        'X18': [0, 1],
        'X19': [0, 1],
        'X20': [0, 1],
        'X21': [0, 1],
        'X22': [0, 1],
        'X23': [0, 1],}
label = {'Y': [0, 1]}




for feature_selection in range(1):
    for max_depth in range(23,24):
        print('m:', feature_selection, ' d:', max_depth)
        # ID3
        dt_generator = dt.ID3(feature_selection=feature_selection, max_depth=max_depth+1)
        # get decision tree
        decision_tree = dt_generator.generate_decision_tree(train_data, features, label)
        # train acc
        # predict
        train_data['py']= dt_generator.classify(decision_tree, train_data)
        acc = train_data.apply(lambda row: 1 if row['Y'] == row['py'] else 0, axis=1).sum() / train_size
        train_err = 1 - acc
        # test acc
        # predict
        test_data['py']= dt_generator.classify(decision_tree, test_data)
        acc= test_data.apply(lambda row: 1 if row['Y'] == row['py'] else 0, axis=1).sum() / test_size
        test_err = 1 - acc
        print('train_err: ', train_err, 'test_err: ', test_err)







