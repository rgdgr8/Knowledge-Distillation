import tensorflow as tf
import pandas as pd
import numpy as np
from math import ceil, log2
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

og = pd.read_excel('Labeled_Training_HCXY_AP_30.xlsx').drop(columns=['ECoord', 'NCoord', 'FloorID', 'BuildingID', 'SceneID', 'UserID', 'PhoneID', 'SampleTimes'])
og_exp = np.array(list(map(lambda x: x-1, og.pop('Label'))))
train, test, train_exp, test_exp = train_test_split(og, og_exp, stratify=og_exp, test_size=0.1, random_state=1)
n_classes = len(np.unique(og_exp))
n_inputs = train.shape[1]
print(min(og_exp), max(og_exp), n_classes, n_inputs, train.shape, test.shape)

train = MinMaxScaler().fit_transform(train.values)
test = MinMaxScaler().fit_transform(test.values)

h1_nodes = ceil(pow(2,log2(n_inputs+n_classes)))
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(n_inputs,)))
model.add(tf.keras.layers.Dense(h1_nodes*4, activation='relu'))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model = tf.keras.models.load_model('140_hcxy_level0_classifier')

EPOCHS = 140
model.fit(train, train_exp, epochs=EPOCHS, validation_data=(test, test_exp), shuffle=True, verbose=2)
print('\nEvaluation:', model.evaluate(test, test_exp))
# model.save('140_hcxy_level0_classifier')