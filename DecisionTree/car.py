#%%
import DT as dt
import pandas as pd


# load data
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
types = {'buying': str, 'maint': str, 'doors': str, 'persons': str, 'lug_boot': str, 'safety': str, 'label': str}
# load train data
train_data =  pd.read_csv('../data/car/train.csv', names=columns, dtype=types)
train_size = len(train_data.index)
# load test data
test_data =  pd.read_csv('../data/car/test.csv', names=columns, dtype=types)
test_size = len(test_data.index)
# set features and label 
features = {'buying': ['vhigh', 'high', 'med', 'low'], 
            'maint':  ['vhigh', 'high', 'med', 'low'], 
            'doors':  ['2', '3', '4', '5more'], 
            'persons': ['2', '4', 'more'], 
            'lug_boot': ['small', 'med', 'big'],  
            'safety':  ['low', 'med', 'high']  }

label = {'label': ['unacc', 'acc', 'good', 'vgood']}

train_acc = [[0 for x in range(6)] for y in range(3)]
test_acc = [[0 for x in range(6)] for y in range(3)]


for feature_selection in range(3):
    for max_depth in range(6):
        # ID3
        dt_generator = dt.ID3(feature_selection=feature_selection, max_depth=max_depth+1)
        # get decision tree
        decision_tree = dt_generator.generate_decision_tree(train_data, features, label)
        # train acc
        # predict
        train_data['plabel']= dt_generator.classify(decision_tree, train_data)
        train_acc[feature_selection][max_depth] = train_data.apply(lambda row: 1 if row['label'] == row['plabel'] else 0, axis=1).sum() / train_size
        # test acc
        # predict
        test_data['plabel']= dt_generator.classify(decision_tree, test_data)
        test_acc[feature_selection][max_depth] = test_data.apply(lambda row: 1 if row['label'] == row['plabel'] else 0, axis=1).sum() / test_size

print('train_acc:', train_acc)
print('test_acc:', test_acc)



