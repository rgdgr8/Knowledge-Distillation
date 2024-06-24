import tensorflow as tf
import pandas as pd
import numpy as np
from math import ceil, log2
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
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
    if(i not in data_to_class):
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

train = MinMaxScaler().fit_transform(train.values)
test = MinMaxScaler().fit_transform(test.values)

# from collections import Counter
# from imblearn.over_sampling import SMOTE
# train, train_exp = SMOTE(random_state=212, k_neighbors=min(Counter(train_exp).values())-1).fit_resample(train, train_exp)
# print(train.shape)

#train, train_val, train_exp, train_val_exp = train_test_split(train, train_exp, test_size=0.1, random_state=1)

h1_nodes = ceil(pow(2,log2(n_inputs+n_classes)))
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(n_inputs,)))
model.add(tf.keras.layers.Dropout(0.05))
model.add(tf.keras.layers.Dense(h1_nodes*4, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

EPOCHS = 1300
class_weights = dict(zip(np.unique(train_exp), class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(train_exp), y=train_exp)))
history = model.fit(train, train_exp, epochs=EPOCHS, validation_data=(test, test_exp), class_weight=class_weights, shuffle=True, verbose=2)

#print(history.history)
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
print('\nEvaluation:', model.evaluate(test, test_exp))

model.save('room_level0_classifier')