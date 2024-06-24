import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

og = pd.read_excel('Nexus5_RSSI_data.xlsx').drop(columns=['X', 'Y'])
og_exp = np.array(list(map(lambda x: x-1, og.pop('Label'))))
train, test, train_exp, test_exp = train_test_split(og, og_exp, test_size=0.1, stratify=og_exp, random_state=1)

# train = pd.read_csv(
#     '/Users/rgdgr8/Documents/JUIndoorLoc-Training-data.csv').drop(columns=['Rs', 'Ts', 'Did', 'Hpr'])
# test = pd.read_csv(
#     '/Users/rgdgr8/Documents/JUIndoorLoc-Test-data.csv').drop(columns=['Rs', 'Ts', 'Did', 'Hpr'])
# train_exp = train.pop('Cid')
# test_exp = test.pop('Cid')

# train = pd.read_excel('/Users/rgdgr8/Documents/Train-classroom.xlsx').drop(columns=['CELLID', 'TIMESTAMP', 'DEVICEID'])
# test = pd.read_excel('/Users/rgdgr8/Documents/Test-classroom.xlsx').drop(columns=['CELLID', 'TIMESTAMP', 'DEVICEID'])
# train_exp = train.pop('CLASS')
# test_exp = test.pop('CLASS')

# n_inputs = len(train.keys())
# data_to_class = {}
# n_classes = 0
# for i in train_exp:
#     if(i not in data_to_class):
#         data_to_class[i] = n_classes
#         n_classes += 1
# train_exp = train_exp.apply(lambda x: data_to_class[x])
# test_exp = test_exp.apply(lambda x: data_to_class[x])

print(min(Counter(train_exp).values()),max(Counter(train_exp).values()), len(train_exp))

#######Scale for nexus dataset not for juindoorloc dataset
train = MinMaxScaler().fit_transform(train.values)
test = MinMaxScaler().fit_transform(test.values)

# KNeighborsClassifier(n_neighbors=2, weights='distance') for juindoorloc and nexus
model = KNeighborsClassifier()
tuner = GridSearchCV(model, {'n_neighbors':[2,3,4,5,6,7,8,9,10,11,12,13,14,15], 'weights':['uniform', 'distance']})
est = tuner.fit(train, train_exp).best_estimator_
print(est, tuner.best_score_, tuner.best_params_)
print(est.score(test,test_exp))