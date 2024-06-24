import tensorflow as tf
import pandas as pd
import numpy as np
from math import ceil, log2
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import class_weight
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier

train = pd.read_csv(
    '/Users/rgdgr8/Documents/JUIndoorLoc-Training-data.csv').drop(columns=['Rs', 'Ts', 'Hpr'])
test = pd.read_csv(
    '/Users/rgdgr8/Documents/JUIndoorLoc-Test-data.csv').drop(columns=['Rs', 'Ts', 'Hpr'])
train_exp = train.pop('Cid')
test_exp = test.pop('Cid')

data_to_class = {}
n_classes = 0
for i in train_exp:
    if(i not in data_to_class):
        data_to_class[i] = n_classes
        n_classes += 1
train_exp = train_exp.apply(lambda x: data_to_class[x])
test_exp = test_exp.apply(lambda x: data_to_class[x])

train_did_wise = defaultdict(set)
for i,did in enumerate(train.pop('Did')):
    train_did_wise[did].add(i)
test_did_wise = defaultdict(set)
for i,did in enumerate(test.pop('Did')):
    test_did_wise[did].add(i)

print(train_did_wise.keys())
for did in train_did_wise:
    print(did, ':', len(train_did_wise[did]))
print(test_did_wise.keys())
for did in test_did_wise:
    print(did, ':', len(test_did_wise[did]))

train = MinMaxScaler().fit_transform(train.values)
test = MinMaxScaler().fit_transform(test.values)

n_inputs = len(train[0])
h1_nodes = ceil(pow(2,log2(n_inputs+n_classes)))

def stacked_dataset(members, inputX):
    stackX = None
    for model in members:
        yhat = model.predict(inputX, verbose=0)
        # stack predictions into [rows, members, probabilities]
        if stackX is None:
            stackX = yhat
        else:
            stackX = np.dstack((stackX, yhat))
    # flatten predictions to [rows, members x probabilities]
    stackX = stackX.reshape(
        (stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
    return stackX

def load_all_models(n_models, did):
    all_models = []
    for i in range(n_models):
        filename = did + '_intg_mem' + str(i)
        model = tf.keras.models.load_model(filename)
        all_models.append(model)
    return all_models

for train_did in train_did_wise:
    # model = tf.keras.models.Sequential()
    # model.add(tf.keras.Input(shape=(n_inputs,)))
    # model.add(tf.keras.layers.Dropout(0.05))
    # model.add(tf.keras.layers.Dense(h1_nodes*4, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model = tf.keras.models.load_model(train_did+'_level0_classifier')
    # model = tf.keras.models.load_model(train_did+'_intg_st')
    model = tf.keras.models.load_model('/Users/rgdgr8/Documents/'+train_did+'_sep_st_model')
    members = load_all_models(5, train_did)
    
    current_train = np.array([train[i] for i in train_did_wise[train_did]])
    current_train_exp = np.array([train_exp[i] for i in train_did_wise[train_did]])

    # model = KNeighborsClassifier()
    # tuner = GridSearchCV(model, {'n_neighbors':[2,3,4,5], 'weights':['uniform', 'distance']})
    # est = tuner.fit(current_train, current_train_exp).best_estimator_

    for test_did in test_did_wise:
        current_test = np.array([test[i] for i in test_did_wise[test_did]])
        current_test_exp = np.array([test_exp[i] for i in test_did_wise[test_did]])

        # Y = [current_test for _ in range(len(model.input))]
        Y = stacked_dataset(members, current_test)

        print('----------------------------', train_did, test_did, current_train.shape, current_train_exp.shape, current_test.shape, current_test_exp.shape, model.evaluate(Y, current_test_exp), '----------------------------------')