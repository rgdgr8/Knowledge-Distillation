import tensorflow as tf
import pandas as pd
import numpy as np
from math import ceil, log2
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import class_weight, resample

og = pd.read_excel('Nexus5_RSSI_data.xlsx').drop(columns=['X', 'Y'])
og_exp = np.array(list(map(lambda x: x-1, og.pop('Label'))))
train, test, train_exp, test_exp = train_test_split(og, og_exp, test_size=0.1, stratify=og_exp, random_state=1)
n_classes = len(np.unique(og_exp))

# train = pd.read_csv(
#     '/Users/rgdgr8/Documents/JUIndoorLoc-Training-data.csv').drop(columns=['Rs', 'Ts', 'Did', 'Hpr'])
# test = pd.read_csv(
#     '/Users/rgdgr8/Documents/JUIndoorLoc-Test-data.csv').drop(columns=['Rs', 'Ts', 'Did', 'Hpr'])
# train_exp = train.pop('Cid')
# test_exp = test.pop('Cid')

# train = pd.read_excel('/Users/rgdgr8/Documents/Train-classroom.xlsx').drop(columns=['CELLID', 'TIMESTAMP', 'DEVICEID'])
# test = pd.read_excel('/Users/rgdgr8/Documents/Test-classroom.xlsx').drop(columns=['CELLID', 'TIMESTAMP', 'DEVICEID'])
# train_exp = train.pop('CLASS')
# test_exp = test.pop('CLASS')

n_inputs = len(train.keys())
# data_to_class = {}
# n_classes = 0
# for i in train_exp:
#     if (i not in data_to_class):
#         data_to_class[i] = n_classes
#         n_classes += 1
# train_exp = train_exp.apply(lambda x: data_to_class[x])
# test_exp = test_exp.apply(lambda x: data_to_class[x])

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

n_members = 6
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

# define stacked model from multiple member input models
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
    inpdropout = tf.keras.layers.Dropout(0.1)(merge) #0.2 for juindoorloc
    hidden = tf.keras.layers.Dense(ceil(pow(2,log2(n_members*n_classes)))*4, activation='relu')(inpdropout)
    dropout = tf.keras.layers.Dropout(0.5)(hidden)
    output = tf.keras.layers.Dense(n_classes, activation='softmax')(dropout)
    model = tf.keras.models.Model(inputs=ensemble_visible, outputs=output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def plot_score(history, score, epochs):
    score_train = history.history[score]
    score_val = history.history['val_'+score]
    epochs = range(1,epochs+1)
    plt.plot(epochs, score_train, 'g', label='Training '+score)
    plt.plot(epochs, score_val, 'b', label='validation '+score)
    plt.title('Training and Validation '+score)
    plt.xlabel('Epochs')
    plt.ylabel(score)
    plt.legend()
    plt.show()

class_weights = dict(zip(np.unique(train_exp), class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(train_exp), y=train_exp)))
# fit a stacked model
def fit_stacked_model(model, inputX, inputy, valX, valy):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    Y = [valX for _ in range(len(model.input))]
    EPOCHS = 5
    history = model.fit(X, inputy, validation_data=(Y, valy), epochs=EPOCHS, verbose=2)
    plot_score(history, 'loss', EPOCHS)
    plot_score(history, 'accuracy', EPOCHS)

_, train, __, train_exp = train_test_split(train, train_exp, test_size=0.1, stratify=train_exp, random_state=1)
train_exp = np.array(train_exp)

members = load_all_models(n_members, 'nexus_kfold_ensemble')
# print('Loaded %d models' % len(members))
# define ensemble model
# stacked_model = tf.keras.models.load_model('intg_st_model')
# stacked_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# stacked_model.summary()
stacked_model = define_stacked_model(members)
# fit stacked model on test dataset
fit_stacked_model(stacked_model, train, train_exp, test, test_exp)
stacked_model.save('nexus_intg_st_model')