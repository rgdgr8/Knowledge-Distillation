import tensorflow as tf
import pandas as pd
import numpy as np
from math import ceil, log2
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import class_weight, resample
from sklearn.metrics import accuracy_score
from itertools import product

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

# from scipy import stats
# train[(np.abs(stats.zscore(train)) < 3).all(axis=1)]

# q_low = train["CLASS"].quantile(0.01)
# q_hi  = train["CLASS"].quantile(0.99)
# train = train[(train["CLASS"] < q_hi) & (train["CLASS"] > q_low)] #this step removes the rows as well as their indexes.

# train_exp.hist()
# plt.show()

# train = pd.DataFrame(MinMaxScaler().fit_transform(
#     train.values), index=train.index, columns=train.columns)
# test = pd.DataFrame(MinMaxScaler().fit_transform(
#     test.values), index=test.index, columns=test.columns)
train = MinMaxScaler().fit_transform(train.values)
test = MinMaxScaler().fit_transform(test.values)

# train, _, train_exp, __ = train_test_split(train, train_exp, test_size=0.1, stratify=train_exp, random_state=1)
# train_exp = np.array(train_exp)

# from collections import Counter
# from imblearn.over_sampling import SMOTE
# train, train_exp = SMOTE(random_state=212, k_neighbors=min(Counter(train_exp).values())-1).fit_resample(train, train_exp)

# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, weights, testX):
	# make predictions
	yhats = [model.predict(testX, verbose=0) for model in members]
	yhats = np.array(yhats)
	# weighted sum across ensemble members
	summed = np.tensordot(yhats, weights, axes=((0),(0)))
	# argmax across classes
	result = np.argmax(summed, axis=1)
	return result

# evaluate a specific number of members in an ensemble
def evaluate_ensemble(members, weights, testX, testy):
	# make prediction
	yhat = ensemble_predictions(members, weights, testX)
	# calculate accuracy
	return accuracy_score(testy, yhat)

# normalize a vector to have unit norm
def normalize(weights):
	# calculate l1 vector norm
	result = np.linalg.norm(weights, 1)
	# check for a vector of all zeros
	if result == 0.0:
		return weights
	# return normalized vector (unit norm)
	return weights / result

def grid_search(members, testX, testy):
    # define weights to consider
    w = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    best_score, best_weights = 0.0, None
    # iterate all possible combinations (cartesian product)
    for weights in product(w, repeat=len(members)):
        # hack, normalize weight vector
        weights = normalize(weights)
        # evaluate weights
        score = evaluate_ensemble(members, weights, testX, testy)
        if score > best_score:
            best_score, best_weights = score, weights
            #print('>%s %.3f' % (best_weights, best_score))
    return list(best_weights)

n_splits = 6
#load ensemble members
members = [tf.keras.models.load_model('kfold_ensemble'+str(i)) for i in range(n_splits)]
# evaluate different numbers of ensembles on hold out set
ensemble_scores = list()
try:
    for i in range(4, n_splits+1):
        weights = grid_search(members[:i], test, test_exp)
        ensemble_score = evaluate_ensemble(members[:i], weights, test, test_exp)
        print('> %d: ensemble=%.3f' % (i, ensemble_score))
        ensemble_scores.append(ensemble_score)
finally:
    # plot score vs number of ensemble members
    x_axis = [i for i in range(4, len(ensemble_scores)+1)]
    #plt.plot(x_axis, scores, marker='o', linestyle='None')
    plt.plot(x_axis, ensemble_scores, marker='o')
    plt.show()