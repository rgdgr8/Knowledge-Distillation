import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.utils import class_weight
from collections import Counter
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv(
    '/Users/rgdgr8/Documents/JUIndoorLoc-Training-data.csv').drop(columns=['Rs', 'Ts', 'Did', 'Hpr']).dropna()
test = pd.read_csv(
    '/Users/rgdgr8/Documents/JUIndoorLoc-Test-data.csv').drop(columns=['Rs', 'Ts', 'Did', 'Hpr']).dropna()
train_exp = train.pop('Cid')
test_exp = test.pop('Cid')

# train = pd.read_excel('/Users/rgdgr8/Documents/Train-classroom.xlsx').drop(columns=['CELLID', 'TIMESTAMP', 'DEVICEID'])
# test = pd.read_excel('/Users/rgdgr8/Documents/Test-classroom.xlsx').drop(columns=['CELLID', 'TIMESTAMP', 'DEVICEID'])
# train_exp = train.pop('CLASS')
# test_exp = test.pop('CLASS')

n_inputs = len(train.keys())
data_to_class = {}
n_classes = 0
for i in train_exp:
    if(i not in data_to_class):
        data_to_class[i] = n_classes
        n_classes += 1
train_exp = train_exp.apply(lambda x: data_to_class[x])
test_exp = test_exp.apply(lambda x: data_to_class[x])

print(min(Counter(train_exp).values()),max(Counter(train_exp).values()), len(train_exp))

# train = MinMaxScaler().fit_transform(train.values)
# test = MinMaxScaler().fit_transform(test.values)

model = DecisionTreeClassifier()
tuner = GridSearchCV(model, {'criterion':['gini', 'entropy', 'log_loss'], 'splitter':['best', 'random'], 'max_features':[None, 'sqrt', 'log2'], 'class_weight':['balanced', None]})
est = tuner.fit(train, train_exp).best_estimator_
print(est, tuner.best_score_, tuner.best_params_)
print(est.score(test,test_exp))
# model.fit(train,train_exp)
# print(model.score(test,test_exp))
