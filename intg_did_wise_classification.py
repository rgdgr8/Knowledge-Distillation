import tensorflow as tf
import pandas as pd
import numpy as np
from math import ceil, log2
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import class_weight, resample
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
n_members = 5

def load_all_models(n_models, did, train=[], train_exp=[]):
    all_models = []
    if(len(train)):
        epochs = {'D1': 60, 'D2': 35, 'D3': 35, 'D4': 55}
        kfold = KFold(n_models, shuffle=True, random_state=1)
        for train_ix, test_ix in kfold.split(train):
            trainX, trainy = train[train_ix], train_exp[train_ix]
            testX, testy = train[test_ix], train_exp[test_ix]
            model = tf.keras.models.Sequential()
            model.add(tf.keras.Input(shape=(n_inputs,)))
            model.add(tf.keras.layers.Dropout(0.05))
            model.add(tf.keras.layers.Dense(h1_nodes*4, activation='relu'))
            model.add(tf.keras.layers.Dropout(0.5))
            model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(trainX, trainy, epochs=epochs[did], validation_data=(testX, testy), verbose=2)
            model.save(did+'_intg_mem'+str(len(all_models)))
            all_models.append(model)
    else:
        for i in range(n_models):
            filename = did + '_intg_mem' + str(i)
            model = tf.keras.models.load_model(filename)
            all_models.append(model)
    return all_models

def define_stacked_model(members):
    # update all layers in all models to not be trainable
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers:
            # make not trainable
            layer.trainable = False
            # rename to avoid 'unique layer name' issue
            layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
    # define multi-headed input
    ensemble_visible = [model.input for model in members]
    # concatenate merge output from each model
    ensemble_outputs = [model.output for model in members]
    merge = tf.keras.layers.concatenate(ensemble_outputs)
    inpdropout = tf.keras.layers.Dropout(0.2)(merge)
    hidden = tf.keras.layers.Dense(ceil(pow(2,log2(n_members*n_classes)))*4, activation='relu')(inpdropout)
    dropout = tf.keras.layers.Dropout(0.5)(hidden)
    # hidden2 = tf.keras.layers.Dense(ceil(pow(2,log2(n_members*n_classes))), activation='relu')(dropout)
    # dropout2 = tf.keras.layers.Dropout(0.5)(hidden2)
    output = tf.keras.layers.Dense(n_classes, activation='softmax')(dropout)
    model = tf.keras.models.Model(inputs=ensemble_visible, outputs=output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def fit_stacked_model(model, inputX, inputy, valX, valy):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    Y = [valX for _ in range(len(model.input))]
    EPOCHS = 4 # after 8 epochs 
    model.fit(X, inputy, validation_data=(Y, valy), epochs=EPOCHS, verbose=2)

for train_did in train_did_wise:
    mem_train, current_train, mem_train_exp, current_train_exp = map(np.array, train_test_split([train[i] for i in train_did_wise[train_did]], [train_exp[i] for i in train_did_wise[train_did]], test_size=0.1, random_state=1))
    current_test = []
    current_test_exp = []
    for test_did in test_did_wise:
        if(test_did!=train_did):
            current_test.extend([test[i] for i in test_did_wise[test_did]])
            current_test_exp.extend([test_exp[i] for i in test_did_wise[test_did]])
    current_test = np.array(current_test)
    current_test_exp = np.array(current_test_exp)
    print(train_did, current_train.shape, current_train_exp.shape, current_test.shape, current_test_exp.shape, '-----------------------------------------------------------------')

    #members = load_all_models(n_members, train_did, [], [])
    #stacked_model = define_stacked_model(members)
    stacked_model = tf.keras.models.load_model(train_did+'_intg_st')
    fit_stacked_model(stacked_model, current_train, current_train_exp, current_test, current_test_exp)
    stacked_model.save(train_did+'_intg_st')