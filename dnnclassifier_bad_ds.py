import tensorflow as tf
import sys
import pandas as pd
import numpy as np
from math import ceil, log2
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from collections import Counter

train = pd.read_csv("C:\\Users\\user\\Downloads\\trainingData2.csv")
test = pd.read_csv("C:\\Users\\user\\Downloads\\validationData2.csv")
train_exp = train['BUILDINGID'].astype(str) + '-' + train['FLOOR'].astype(str) + '-' + train['SPACEID'].astype(str)
test_exp = test['BUILDINGID'].astype(str) + '-' + test['FLOOR'].astype(str) + '-' + test['SPACEID'].astype(str)
train = train.drop(columns=['PHONEID', 'TIMESTAMP', 'USERID', 'BUILDINGID', 'FLOOR', 'SPACEID', 'RELATIVEPOSITION', 'LONGITUDE', 'LATITUDE'])
test = test.drop(columns=['PHONEID', 'TIMESTAMP', 'USERID', 'BUILDINGID', 'FLOOR', 'SPACEID', 'RELATIVEPOSITION', 'LONGITUDE', 'LATITUDE'])
# labels = Counter(train_exp)
# for i in range(len(test_exp)):
#     if(labels[test_exp[i]]<2):
#         labels[test_exp[i]] += 1
#         train.loc[len(train)] = test.loc[i]
#         train_exp.loc[len(train_exp)] = test_exp[i]
#         test = test.drop(i)
#         test_exp = test_exp.drop(i)

# train = pd.read_csv("C:\\Users\\user\\Downloads\\JUIndoorLoc-Training-data.csv").drop(columns=['Rs', 'Ts', 'Did', 'Hpr'])
# test = pd.read_csv("C:\\Users\\user\\Downloads\\JUIndoorLoc-Test-data.csv").drop(columns=['Rs', 'Ts', 'Did', 'Hpr'])
# train_exp = train.pop('Cid')
# test_exp = test.pop('Cid')

# train = pd.read_excel('/Users/rgdgr8/Documents/Train-classroom.xlsx').drop(columns=['CELLID', 'TIMESTAMP', 'DEVICEID'])
# test = pd.read_excel('/Users/rgdgr8/Documents/Test-classroom.xlsx').drop(columns=['CELLID', 'TIMESTAMP', 'DEVICEID'])
# train_exp = train.pop('CLASS')
# test_exp = test.pop('CLASS')

# out = sys.stdout
# sys.stdout = open("C:\\Users\\user\\Downloads\\x.txt","w")

n_inputs = len(train.keys())
data_to_class = {}
n_classes = 0
for i in np.concatenate((train_exp,test_exp)):
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

train = pd.DataFrame(MinMaxScaler().fit_transform(train.values), index=train.index, columns=train.columns)
test = pd.DataFrame(MinMaxScaler().fit_transform(test.values), index=test.index, columns=test.columns)

# from imblearn.over_sampling import SMOTE
# train, train_exp = SMOTE(random_state=212, k_neighbors=min(Counter(train_exp).values())-1).fit_resample(train, train_exp)
# print(train.shape)

#train, train_val, train_exp, train_val_exp = train_test_split(train, train_exp, test_size=0.1, random_state=1)

h1_nodes = ceil(pow(2,log2(n_inputs+n_classes)))
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(n_inputs,)))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(h1_nodes*4, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(h1_nodes*2, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(h1_nodes, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

EPOCHS = 50
#class_weights = dict(zip(np.unique(train_exp), class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(train_exp), y=train_exp)))
history = model.fit(train, train_exp, epochs=EPOCHS, validation_data=(test, test_exp), shuffle=True, verbose=2)
model.save('weighted_avg_minimal_model')
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1,EPOCHS+1)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#model = tf.keras.models.load_model('weighted_avg')
print('\nEvaluation:', model.evaluate(test, test_exp))