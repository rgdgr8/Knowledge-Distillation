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
n_members = 5

def load_all_models(n_models, did):
    all_models = []
    for i in range(n_models):
        filename = did + '_intg_mem' + str(i) #the member names ARE did+_intg_mem_+i for both sep_st and intg_st in deviceid-wise classification
        model = tf.keras.models.load_model(filename)
        all_models.append(model)
    return all_models

# create stacked model input dataset as outputs from the ensemble
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

def fit_stacked_model(did, members, inputX, inputy, valX, valy):
    X = stacked_dataset(members, inputX)
    Y = stacked_dataset(members, valX)
    model = tf.keras.models.load_model('/Users/rgdgr8/Documents/'+did+'_sep_st_model')
    # h1_nodes = len(members)*n_classes
    # model = tf.keras.models.Sequential()
    # model.add(tf.keras.Input(shape=(h1_nodes,)))
    # model.add(tf.keras.layers.Dropout(0.2))
    # model.add(tf.keras.layers.Dense(h1_nodes*4, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    EPOCHS = 5 # after 10
    model.fit(X, inputy, validation_data=(Y, valy), epochs=EPOCHS, verbose=2)
    #print(model.evaluate(Y, valy))
    return model

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

    members = load_all_models(n_members, train_did)
    fit_stacked_model(train_did, members, current_train, current_train_exp, current_test, current_test_exp)
    # .save('/Users/rgdgr8/Documents/'+train_did+'_sep_st_model')