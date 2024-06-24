import tensorflow as tf
import pandas as pd
import numpy as np
from math import ceil, log2
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from collections import defaultdict

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
EPOCHS = 20
for train_did in train_did_wise:
    model = tf.keras.models.load_model(train_did+'_level0_classifier')
    # model = tf.keras.models.Sequential()
    # model.add(tf.keras.Input(shape=(n_inputs,)))
    # model.add(tf.keras.layers.Dropout(0.05))
    # model.add(tf.keras.layers.Dense(h1_nodes*4, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    current_train = np.array([train[i] for i in train_did_wise[train_did]])
    current_train_exp = np.array([train_exp[i] for i in train_did_wise[train_did]])
    current_test = []
    current_test_exp = []
    for test_did in test_did_wise:
        if(test_did!=train_did):
            current_test.extend([test[i] for i in test_did_wise[test_did]])
            current_test_exp.extend([test_exp[i] for i in test_did_wise[test_did]])
    current_test = np.array(current_test)
    current_test_exp = np.array(current_test_exp)
    print(train_did, current_train.shape, current_train_exp.shape, current_test.shape, current_test_exp.shape, model.evaluate(current_test, current_test_exp), '-----------------------------------------------------------------')

    # model.fit(current_train, current_train_exp, epochs=EPOCHS, validation_data=(current_test, current_test_exp), shuffle=True, verbose=2)
    # model.save(train_did+'_level0_classifier')