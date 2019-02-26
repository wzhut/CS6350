#%%
import pandas as pd
import numpy as np

data = pd.read_csv('./credit.csv', header=None)

sample = data.sample(frac=1, replace=False)
train_data = sample.iloc[:24000]
test_data = sample.iloc[24000:]

train_data.to_csv('./train.csv', header=None, index=False)
test_data.to_csv('./test.csv', header=None, index=False)
