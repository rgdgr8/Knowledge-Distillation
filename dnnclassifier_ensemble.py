import tensorflow as tf
import pandas as pd
import numpy as np
from math import ceil, log2
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import class_weight, resample
from sklearn.metrics import accuracy_score

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

train, _, train_exp, __ = train_test_split(train, train_exp, test_size=0.1, stratify=train_exp, random_state=1)
train_exp = np.array(train_exp)

h1_nodes = ceil(pow(2, log2(n_inputs+n_classes)))
def evaluate_model(trainX, trainy, testX, testy):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(n_inputs,)))
    model.add(tf.keras.layers.Dropout(0.05))
    model.add(tf.keras.layers.Dense(h1_nodes*4, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
    # model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #class_weights = dict(zip(sorted_labels, class_weight.compute_class_weight(class_weight="balanced", classes=sorted_labels, y=trainy)))
    model.fit(trainX, trainy, epochs=25, verbose=2)
    _, test_acc = model.evaluate(testX, testy, verbose=0)
    return model, test_acc

# def evaluate_model(i, testX, testy):
#     model = tf.keras.models.load_model('kfold_ensemble'+str(i))
#     _, test_acc = model.evaluate(testX, testy, verbose=0)
#     return model, test_acc

def ensemble_predictions(members, testX):
	yhats = [model.predict(testX) for model in members]
	yhats = np.array(yhats)
	# sum across ensemble members
	summed = np.sum(yhats, axis=0)
	# argmax across classes
	result = np.argmax(summed, axis=1)
	return result

# evaluate a specific number of members in an ensemble
def evaluate_n_members(members, n_members, testX, testy):
	# select a subset of members
	subset = members[:n_members]
	# make prediction
	yhat = ensemble_predictions(subset, testX)
	# calculate accuracy
	return accuracy_score(testy, yhat)

n_splits = 10
scores, members = list(), list()
# ix = [i for i in range(len(train))]
# for _ in range(n_splits):
    # select indexes
    # train_ix = resample(ix, replace=True, n_samples=int(len(train)*0.9))
    # test_ix = [x for x in ix if x not in train_ix]
kfold = KFold(n_splits, shuffle=True, random_state=1)
for train_ix, test_ix in kfold.split(train):
    # select data
    trainX, trainy = train[train_ix], train_exp[train_ix]
    # testX, testy = train[test_ix], train_exp[test_ix]
    # evaluate model
    model, test_acc = evaluate_model(trainX, trainy, test, test_exp)
    # model, test_acc = evaluate_model(len(members), test, test_exp)
    print('>**********************%.3f**********************' % test_acc)
    scores.append(test_acc)
    members.append(model)
# summarize expected performance
print('Estimated Accuracy %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

# evaluate different numbers of ensembles on hold out set
ensemble_scores = list()
for i in range(1, n_splits+1):
    ensemble_score = evaluate_n_members(members, i, test, test_exp)
    print('> %d: single=%.3f, ensemble=%.3f' %
          (i, scores[i-1], ensemble_score))
    ensemble_scores.append(ensemble_score)

# plot score vs number of ensemble members
x_axis = [i for i in range(1, n_splits+1)]
plt.plot(x_axis, scores, marker='o', linestyle='None')
plt.plot(x_axis, ensemble_scores, marker='o')
plt.show()

for i, model in enumerate(members):
    model.save('kfold_ensemble'+str(i))
