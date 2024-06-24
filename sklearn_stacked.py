import tensorflow as tf
import pandas as pd
import numpy as np
from math import ceil, log2
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import class_weight
from sklearn.ensemble import StackingClassifier
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

n_members = 5

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
# def create_model1(inputs=n_classes*n_members):
#     h1_nodes = ceil(pow(2,log2(inputs+n_classes)))
#     model = tf.keras.models.Sequential()
#     model.add(tf.keras.Input(shape=(inputs,)))
#     model.add(tf.keras.layers.Dropout(0.1))
#     model.add(tf.keras.layers.Dense(h1_nodes*4, activation='relu'))
#     model.add(tf.keras.layers.Dropout(0.5))
#     model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
#     #model.summary()
#     #model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     return model

for i in range(1,21):
    level0_models = [('base'+str(_), KerasClassifier(model=create_model0, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], epochs=200, verbose=0)) for _ in range(n_members)]
    level1_model = LinearSVC(dual=False, class_weight='balanced', max_iter=i, verbose=False)
    stacked_model = StackingClassifier(estimators=level0_models, final_estimator=level1_model, cv=2, verbose=2)
    stacked_model.fit(train, train_exp)
    print('*********************', i, ':', stacked_model.score(test, test_exp), '*********************')

# for i,model in enumerate(stacked_model.estimators_):
#     model.model_.save('level0_kfold'+str(i))
#stacked_model.final_estimator_.model_.save('level1')