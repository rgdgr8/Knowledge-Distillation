import tensorflow as tf
import pandas as pd
import numpy as np
from math import ceil, log2
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import class_weight
from sklearn.ensemble import AdaBoostClassifier
from scikeras.wrappers import KerasClassifier

# train = pd.read_csv(
#     '/Users/rgdgr8/Documents/JUIndoorLoc-Training-data.csv').drop(columns=['Rs', 'Ts', 'Did', 'Hpr'])
# test = pd.read_csv(
#     '/Users/rgdgr8/Documents/JUIndoorLoc-Test-data.csv').drop(columns=['Rs', 'Ts', 'Did', 'Hpr'])
# train_exp = train.pop('Cid')
# test_exp = test.pop('Cid')

train = pd.read_excel('/Users/rgdgr8/Documents/Train-classroom.xlsx').drop(columns=['CELLID', 'TIMESTAMP', 'DEVICEID'])
test = pd.read_excel('/Users/rgdgr8/Documents/Test-classroom.xlsx').drop(columns=['CELLID', 'TIMESTAMP', 'DEVICEID'])
train_exp = train.pop('CLASS')
test_exp = test.pop('CLASS')

n_inputs = len(train.keys())
data_to_class = {}
n_classes = 0
for i in train_exp:
    if(i not in data_to_class):
        data_to_class[i] = n_classes
        n_classes += 1
train_exp = train_exp.apply(lambda x: data_to_class[x])
test_exp = test_exp.apply(lambda x: data_to_class[x])

train = pd.DataFrame(MinMaxScaler().fit_transform(train.values), index=train.index, columns=train.columns)
test = pd.DataFrame(MinMaxScaler().fit_transform(test.values), index=test.index, columns=test.columns)

def create_model0(inputs=n_inputs):
    h1_nodes = ceil(pow(2,log2(inputs+n_classes)))
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(inputs,)))
    model.add(tf.keras.layers.Dropout(0.05))
    model.add(tf.keras.layers.Dense(h1_nodes*4, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
    #model.summary()
    #model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

basemodel = KerasClassifier(model=create_model0, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], epochs=1000, verbose=2)
model = AdaBoostClassifier(base_estimator=basemodel)
model.fit(train, train_exp)
print(model.score(test,test_exp))
scores = []
for i,est in enumerate(model.estimators_):
    scores.append(est.score(test, test_exp))

# plot score vs number of ensemble members
x_axis = [i for i in range(1, len(scores)+1)]
plt.plot(x_axis, scores, marker='o')
plt.show()