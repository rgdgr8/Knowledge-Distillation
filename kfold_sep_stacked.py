import tensorflow as tf
import pandas as pd
import numpy as np
from math import ceil, log2
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.utils import class_weight

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
    if (i not in data_to_class):
        data_to_class[i] = n_classes
        n_classes += 1
train_exp = train_exp.apply(lambda x: data_to_class[x])
test_exp = test_exp.apply(lambda x: data_to_class[x])

train = MinMaxScaler().fit_transform(train.values)
test = MinMaxScaler().fit_transform(test.values)

def create_model(n_inputs, input_dropout):
    h1_nodes = ceil(pow(2,log2(n_inputs+n_classes)))
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(n_inputs,)))
    model.add(tf.keras.layers.Dropout(input_dropout))
    model.add(tf.keras.layers.Dense(h1_nodes*4, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def fit_model(model, trainX, trainy, testX=[]):
    model.fit(trainX, trainy, epochs=200, verbose=2)
    if(len(testX)>0):
        return model.predict(testX)

n_splits = 5 #this must be greater than 1
members, predictions, level1_test = [], None, None
for _ in range(n_splits):
    predx, ix = [], []
    kfold = KFold(3, shuffle=True)
    for train_ix, test_ix in kfold.split(train):
        trainX, trainy = train[train_ix], train_exp[train_ix]
        testX, testy = train[test_ix], train_exp[test_ix]
        prediction = fit_model(create_model(n_inputs, 0.05), trainX, trainy, testX)
        ix.extend(test_ix)
        predx.extend(prediction)
    model = create_model(n_inputs, 0.05)
    fit_model(model, train, train_exp)
    members.append(model)
    predx = [x for _,x in sorted(zip(ix,predx))]
    try:
        predictions = np.concatenate((predictions, predx), axis=1)
        level1_test = np.concatenate((level1_test, model.predict(test)), axis=1)
    except:
        predictions = predx
        level1_test = model.predict(test)
print(n_classes, train.shape, predictions.shape, test.shape, level1_test.shape)

# pd.DataFrame(predictions).to_csv('stack_train.csv')
# pd.DataFrame(level1_test).to_csv('stack_test.csv')
pd.DataFrame(predictions).to_csv('room_stack_train.csv')
pd.DataFrame(level1_test).to_csv('room_stack_test.csv')
################################ ADD index=False in to_csv if you dont want extra column in csv file #############################

for i, model in enumerate(members):
    # model.save('proper_kfold_base'+str(i))
    model.save('room_proper_kfold_base'+str(i))

EPOCHS = 20
level1_model = create_model(n_classes*n_splits, 0.2)
level1_model.summary()
class_weights = dict(zip(np.unique(train_exp), class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(train_exp), y=train_exp)))
history = level1_model.fit(predictions, train_exp, class_weight=class_weights, validation_data=(level1_test, test_exp), epochs=EPOCHS, verbose=2)

# level1_model.save('proper_kfold_sep_stacked')
level1_model.save('room_proper_kfold_sep_stacked')

def plot_score(score, epochs):
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

plot_score('loss', EPOCHS)
plot_score('accuracy', EPOCHS)
print('\nEvaluation:', level1_model.evaluate(level1_test, test_exp))