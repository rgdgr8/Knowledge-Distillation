import tensorflow as tf
import pandas as pd
import numpy as np
from math import ceil, log2
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import class_weight, resample
from sklearn.ensemble import VotingClassifier
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv(
    '/Users/rgdgr8/Documents/JUIndoorLoc-Training-data.csv').drop(columns=['Rs', 'Ts', 'Did', 'Hpr'])
test = pd.read_csv(
    '/Users/rgdgr8/Documents/JUIndoorLoc-Test-data.csv').drop(columns=['Rs', 'Ts', 'Did', 'Hpr'])
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
    if (i not in data_to_class):
        data_to_class[i] = n_classes
        n_classes += 1
train_exp = train_exp.apply(lambda x: data_to_class[x])
test_exp = test_exp.apply(lambda x: data_to_class[x])

train = MinMaxScaler().fit_transform(train.values)
test = MinMaxScaler().fit_transform(test.values)

h1_nodes = ceil(pow(2, log2(n_inputs+n_classes)))
def define_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(n_inputs,)))
    model.add(tf.keras.layers.Dropout(0.05))
    model.add(tf.keras.layers.Dense(h1_nodes*4, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
    # model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

n_splits = 2
members = list()
# ix = [i for i in range(len(train))]
# for _ in range(n_splits):
    # select indexes
    # train_ix = resample(ix, replace=True, n_samples=int(len(train)*0.9))
    # test_ix = [x for x in ix if x not in train_ix]
kfold = KFold(n_splits, shuffle=True, random_state=1)
for train_ix, test_ix in kfold.split(train):
    trainX, trainy = train[train_ix], train_exp[train_ix]
    members.append(('base'+str(len(members)), KerasClassifier(model=define_model, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], epochs=1, verbose=2)))
    members[-1][1].fit(trainX, trainy)

voting_model = VotingClassifier(estimators=members, voting='soft', verbose=True)
voting_model.estimators = [members[i][1] for i in range(len(members))]
voting_model.le_ = LabelEncoder().fit(train_exp)
voting_model.classes_ = voting_model.le_.classes_
voting_model.score(test,test_exp)