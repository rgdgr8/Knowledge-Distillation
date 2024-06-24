import tensorflow as tf
import pandas as pd
import numpy as np
from math import ceil, log2, dist
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from collections import Counter, defaultdict

print(tf.__version__)
# print(tf.keras.__version__)

train = pd.read_csv(
    'JUIndoorLoc-Training-data.csv').drop(columns=['Rs', 'Ts', 'Did', 'Hpr']).dropna()
test = pd.read_csv(
    'JUIndoorLoc-Test-data.csv').drop(columns=['Rs', 'Ts', 'Did', 'Hpr']).dropna()
# print(train.shape)
# print(test.shape)
train_exp = train.pop('Cid')
test_exp = test.pop('Cid')

n_inputs = len(train.keys())
data_to_class = {}
n_classes = 0
for i in train_exp:
    if(i not in data_to_class):
        data_to_class[i] = n_classes
        n_classes += 1
train_exp = train_exp.apply(lambda x: data_to_class[x])
test_exp = test_exp.apply(lambda x: data_to_class[x])

train = MinMaxScaler().fit_transform(train.values)
test = MinMaxScaler().fit_transform(test.values)

class_to_data = {}
x = Counter()
for label in data_to_class:
    coord = tuple(map(int,label[1:].split('-')))
    x[coord[0]] += 1
    class_to_data[data_to_class[label]] = coord

# print(len(data_to_class), len(class_to_data))
print(x)

n_members = 5 #5 for juindoorloc #6 for nexus
# load models from file
def load_all_models(n_models, model_name):
    all_models = list()
    for i in range(n_models):
        filename = model_name + str(i)
        # load model from file
        model = tf.keras.models.load_model(filename)
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models

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

def plot_graph(model, test, line_color, line_label):
    probs = model.predict(test)
    preds = probs.argmax(axis=1)
    #print(len(preds), len(test_exp))

    loc_errs = defaultdict(int)
    fl_diff = []
    for true_class, pred in zip(test_exp,preds):
        pred_coord = class_to_data[pred]
        true_coord = class_to_data[true_class]
        if(pred_coord[0]!=true_coord[0]):
            fl_diff.append(('L'+'-'.join(map(str,pred_coord)), 'L'+'-'.join(map(str, true_coord))))
        else:
            # loc_errs[dist(pred_coord, true_coord)].append('L'+'-'.join(map(str,true_coord)))
            loc_errs[dist(pred_coord, true_coord)] += 1

    print(line_label, len(fl_diff), len(loc_errs))

    loc_errs = pd.DataFrame(loc_errs.items())
    print(loc_errs)
    loc_errs.to_csv('localisation_error_vs_frequency_for_'+line_label+'.csv', header=['Localisation Error', 'Frequency'], index=False)
    
    fl_diff = pd.DataFrame(fl_diff)
    print(fl_diff)
    fl_diff.to_csv('localisation_error_due_to_floor_diff_for_'+ line_label + '.csv', header=['Prediction', 'Expected'], index=False)

    return

    # getting data of the histogram
    count, bins_count = np.histogram(loc_errs, bins=ceil(max(loc_errs)-min(loc_errs)))
    # finding the PDF of the histogram using count values
    pdf = count / sum(count)
    # using numpy np.cumsum to calculate the CDF
    cdf = np.cumsum(pdf)
    # plotting PDF and CDF
    plt.plot(bins_count[1:], cdf, color=line_color, label=line_label)

# plot_graph(tf.keras.models.load_model('level0_classifier'), test, 'b', 'Base')

sep_test = stacked_dataset(load_all_models(n_members, 'kfold_ensemble'), test)
plot_graph(tf.keras.models.load_model('sep_st_model'), sep_test, 'g', 'NN Separate Stacking with 0.1 train test split ')

# intst_model = tf.keras.models.load_model('intg_st_model')
# intst_test = [test for _ in range(len(intst_model.input))]
# plot_graph(intst_model, intst_test, 'r', 'Integrated Stacking')

# plt.legend()
# plt.xlabel("Localisation Error")
# plt.ylabel("Cumulative Distribution Function")
# plt.show()