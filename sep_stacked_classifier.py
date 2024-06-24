import tensorflow as tf
import pandas as pd
import numpy as np
from math import ceil, log2
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import class_weight, resample

# og = pd.read_excel('Nexus5_RSSI_data.xlsx').drop(columns=['X', 'Y'])
# og_exp = np.array(list(map(lambda x: x-1, og.pop('Label'))))
# train, test, train_exp, test_exp = train_test_split(og, og_exp, test_size=0.1, stratify=og_exp, random_state=1)
# n_classes = len(np.unique(og_exp))
# n_inputs = og.shape[1]

og = pd.read_excel('Labeled_Training_HCXY_AP_30.xlsx').drop(columns=['ECoord', 'NCoord', 'FloorID', 'BuildingID', 'SceneID', 'UserID', 'PhoneID', 'SampleTimes'])
og_exp = np.array(list(map(lambda x: x-1, og.pop('Label'))))
train, test, train_exp, test_exp = train_test_split(og, og_exp, stratify=og_exp, test_size=0.1, random_state=1)
n_classes = len(np.unique(og_exp))
n_inputs = train.shape[1]
print(n_classes, n_inputs)

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

# n_inputs = len(train.keys())
# data_to_class = {}
# n_classes = 0
# for i in train_exp:
#     if (i not in data_to_class):
#         data_to_class[i] = n_classes
#         n_classes += 1
# train_exp = train_exp.apply(lambda x: data_to_class[x])
# test_exp = test_exp.apply(lambda x: data_to_class[x])

train = MinMaxScaler().fit_transform(train.values)
test = MinMaxScaler().fit_transform(test.values)

n_members = 10 #5 for juindoorloc #6 for nexus #10 for hcxy
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

def fit_stacked_model(members, inputX, inputy, valX, valy):
    X = stacked_dataset(members, inputX)
    Y = stacked_dataset(members, valX)
    model = tf.keras.models.load_model('hcxy_sep_st_model_smaller')
    print('model reloaded for retraining')
    # h1_nodes = len(members)*n_classes
    # model = tf.keras.models.Sequential()
    # model.add(tf.keras.Input(shape=(h1_nodes,)))
    # # model.add(tf.keras.layers.Dropout(0.1)) #not needed for hcxy
    # model.add(tf.keras.layers.Dense(h1_nodes*4, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    EPOCHS = 10
    history = model.fit(X, inputy, validation_data=(Y, valy), epochs=EPOCHS, verbose=2)
    # plot_score(history, 'loss', EPOCHS)
    # plot_score(history, 'accuracy', EPOCHS)
    return model

_, train, __, train_exp = train_test_split(train, train_exp, test_size=0.1, stratify=train_exp, random_state=1)
train_exp = np.array(train_exp)

members = load_all_models(n_members, 'hcxy_kfold_ensemble')
# fit stacked model on test dataset
model = fit_stacked_model(members, train, train_exp, test, test_exp)

if(input()=='x'):
    model.save('hcxy_sep_st_model_smaller')