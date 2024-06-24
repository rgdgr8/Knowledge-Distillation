import tensorflow as tf
import pandas as pd
import numpy as np
from math import ceil, log2
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

og = pd.read_excel('Nexus5_RSSI_data.xlsx').drop(columns=['X', 'Y'])
og_exp = np.array(list(map(lambda x: x-1, og.pop('Label'))))
og = MinMaxScaler().fit_transform(og.values)
train, test, train_exp, test_exp = train_test_split(og, og_exp, test_size=0.1, stratify=og_exp, random_state=1)
n_classes = len(np.unique(og_exp))

# train = pd.read_csv('nexus_train.csv')
# test = pd.read_csv('nexus_student_train.csv')
# train_exp = train.pop(str(train.shape[1]-1))
# test_exp = test.pop(str(test.shape[1]-1))

######################FOR CREATING DATASET FOR KDN########################################
# train = pd.read_excel('Nexus5_RSSI_data.xlsx').drop(columns=['X', 'Y'])
# train_exp = train.pop('Label')

# n_inputs = len(train.keys())
# data_to_class = {}
# n_classes = 0
# for i in train_exp:
#     if(i not in data_to_class):
#         data_to_class[i] = n_classes
#         n_classes += 1
# train_exp = train_exp.apply(lambda x: data_to_class[x])
# print(train.shape, len(train_exp))

# train = MinMaxScaler().fit_transform(train.values)

# train, test, train_exp, test_exp = train_test_split(train, train_exp, test_size=0.2, random_state=1, stratify=train_exp)
# test, student_test, test_exp, student_test_exp =  train_test_split(test, test_exp, test_size=0.2, random_state=1, stratify=test_exp)
# print(len(train), len(train_exp), len(test), len(test_exp), len(student_test), len(student_test_exp))

# pd.DataFrame(np.column_stack((train, train_exp))).to_csv('nexus_train.csv',index=False)
# pd.DataFrame(np.column_stack((test, test_exp))).to_csv('nexus_student_train.csv',index=False)
# pd.DataFrame(np.column_stack((student_test, student_test_exp))).to_csv('nexus_test.csv',index=False)
################################################################################################

# n_inputs = train.shape[1]
# h1_nodes = ceil(pow(2,log2(n_inputs+n_classes)))
# model = tf.keras.models.Sequential()
# model.add(tf.keras.Input(shape=(n_inputs,)))
# model.add(tf.keras.layers.Dense(h1_nodes*4, activation='relu'))
# model.add(tf.keras.layers.Dense(h1_nodes*2, activation='relu'))
# model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
# model.summary()
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model = tf.keras.models.load_model('nexus_level0_classifier')

EPOCHS = 70    #40 for kdn teacher last tested, might need to be increased due to weird dataset
history = model.fit(train, train_exp, epochs=EPOCHS, validation_data=(test, test_exp), shuffle=True, verbose=2)

# model.save('nexus_teacher')
model.save('nexus_level0_classifier')