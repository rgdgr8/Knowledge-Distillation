# import tensorflow as tf
import pandas as pd
# import numpy as np
# from math import ceil, log2
# from matplotlib import pyplot as plt
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.model_selection import KFold
# from sklearn.utils import class_weight

og1 = pd.read_excel('Labeled_Training_HCXY_AP_30.xlsx')
print('hcxy', og1.shape)
og2 = pd.read_excel('Nexus5_RSSI_data.xlsx')
print('nexus', og2.shape)

# # # n_inputs = len(train.keys())
# # # data_to_class = {}
# # # n_classes = 0
# # # for i in train_exp:
# # #     if(i not in data_to_class):
# # #         data_to_class[i] = n_classes
# # #         n_classes += 1
# # # train_exp = train_exp.apply(lambda x: data_to_class[x])
# # # test_exp = test_exp.apply(lambda x: data_to_class[x])

# # # train = pd.read_csv('stack_train.csv')
# # # test = pd.read_csv('stack_test.csv')
# # # train = np.delete(train.values, 0, axis=1)
# # # test = np.delete(test.values, 0, axis=1)

# # # def create_model(n_inputs, input_dropout):
# # #     h1_nodes = ceil(pow(2,log2(n_inputs+n_classes)))
# # #     model = tf.keras.models.Sequential()
# # #     model.add(tf.keras.Input(shape=(n_inputs,)))
# # #     model.add(tf.keras.layers.Dropout(input_dropout))
# # #     model.add(tf.keras.layers.Dense(h1_nodes*4, activation='relu'))
# # #     model.add(tf.keras.layers.Dropout(0.5))
# # #     model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
# # #     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# # #     return model

# # # # EPOCHS = 22
# # # # level1_model = create_model(len(train[0]), 0.2)
# # # # level1_model.fit(train, train_exp, validation_data=(test, test_exp), epochs=EPOCHS, verbose=2)
# # # # level1_model.evaluate(test,test_exp)
# # # # level1_model.save('proper_kfold_sep_stacked')

# # # # teacher = tf.keras.models.load_model('proper_kfold_sep_stacked')
# # # # teacher.summary()
# # # student = tf.keras.models.load_model('student_test')
# # # student.summary()
# # # print('Student:', student.evaluate(train,train_exp))

# # import tensorflow as tf
# # import pandas as pd
# # # import numpy as np
# # # from math import ceil, log2
# # # from matplotlib import pyplot as plt
# # # from sklearn.preprocessing import StandardScaler, MinMaxScaler
# # # from sklearn.utils import class_weight
# # # from collections import defaultdict

# # og = pd.read_excel('Nexus5_RSSI_data.xlsx')
# # x = pd.read_csv('nexus_train.csv')
# # y = pd.read_csv('nexus_student_train.csv')
# # z = pd.read_csv('nexus_test.csv')

# # print(og.shape, x.shape, y.shape, z.shape)

# # # train = pd.read_csv(
# # #     '/Users/rgdgr8/Documents/JUIndoorLoc-Training-data.csv').drop(columns=['Rs', 'Ts', 'Did', 'Hpr'])
# # # test = pd.read_csv(
# # #     '/Users/rgdgr8/Documents/JUIndoorLoc-Test-data.csv').drop(columns=['Rs', 'Ts', 'Did', 'Hpr'])
# # # train_exp = train.pop('Cid')
# # # test_exp = test.pop('Cid')

# # # # train = pd.read_excel('/Users/rgdgr8/Documents/Train-classroom.xlsx').drop(columns=['CELLID', 'TIMESTAMP', 'DEVICEID'])
# # # # test = pd.read_excel('/Users/rgdgr8/Documents/Test-classroom.xlsx').drop(columns=['CELLID', 'TIMESTAMP', 'DEVICEID'])
# # # # train_exp = train.pop('CLASS')
# # # # test_exp = test.pop('CLASS')

# # # n_inputs = len(train.keys())
# # # data_to_class = {}
# # # n_classes = 0
# # # for i in train_exp:
# # #     if(i not in data_to_class):
# # #         data_to_class[i] = n_classes
# # #         n_classes += 1
# # # train_exp = train_exp.apply(lambda x: data_to_class[x])
# # # test_exp = test_exp.apply(lambda x: data_to_class[x])

# # # train = MinMaxScaler().fit_transform(train.values)
# # # test = MinMaxScaler().fit_transform(test.values)

# # # # train = pd.read_csv('stack_train.csv')
# # # test = pd.read_csv('stack_test.csv')
# # # # train = np.delete(train.values, 0, axis=1)
# # # test = np.delete(test.values, 0, axis=1)

# # # t1 = []
# # # t2 = []
# # # y_t1 = []
# # # y_t2 = []

# # # dups = defaultdict(lambda:[])
# # # for i in range(len(test_exp)):
# # #     dups[test_exp[i]].append(i)
# # # for i in dups.values():
# # #     if(len(i)>1):
# # #         t2.append(test[i[-1]])
# # #         y_t2.append(test_exp[i.pop()])
# # #     for j in i:
# # #         t1.append(test[j])
# # #         y_t1.append(test_exp[j])

# # # t1 = np.array(t1)
# # # y_t1 = np.array(y_t1)
# # # t2 = np.array(t2)
# # # y_t2 = np.array(y_t2)
# # # print('lengths', len(t1), len(t1[0]), len(y_t1), len(t2), len(t2[0]), len(y_t2))

# # class Distiller(tf.keras.Model):
# #     def __init__(self, student, teacher):
# #         super().__init__()
# #         self.teacher = teacher
# #         self.student = student

# #     def compile(
# #         self,
# #         optimizer,
# #         metrics,
# #         student_loss_fn,
# #         distillation_loss_fn,
# #         alpha=0.1,
# #         temperature=3,
# #     ):
# #         """ Configure the distiller.

# #         Args:
# #             optimizer: Keras optimizer for the student weights
# #             metrics: Keras metrics for evaluation
# #             student_loss_fn: Loss function of difference between student
# #                 predictions and ground-truth
# #             distillation_loss_fn: Loss function of difference between soft
# #                 student predictions and soft teacher predictions
# #             alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
# #             temperature: Temperature for softening probability distributions.
# #                 Larger temperature gives softer distributions.
# #         """
# #         super().compile(optimizer=optimizer, metrics=metrics)
# #         self.student_loss_fn = student_loss_fn
# #         self.distillation_loss_fn = distillation_loss_fn
# #         self.alpha = alpha
# #         self.temperature = temperature

# #     def train_step(self, data):
# #         # Unpack data
# #         x, y = data

# #         # Forward pass of teacher
# #         teacher_predictions = self.teacher(x, training=False)

# #         with tf.GradientTape() as tape:
# #             # Forward pass of student
# #             student_predictions = self.student(x, training=True)

# #             # Compute losses
# #             student_loss = self.student_loss_fn(y, student_predictions)

# #             # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
# #             # The magnitudes of the gradients produced by the soft targets scale
# #             # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
# #             distillation_loss = (
# #                 self.distillation_loss_fn(
# #                     tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
# #                     tf.nn.softmax(student_predictions / self.temperature, axis=1),
# #                 )
# #                 * self.temperature**2
# #             )

# #             loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

# #         # Compute gradients
# #         trainable_vars = self.student.trainable_variables
# #         gradients = tape.gradient(loss, trainable_vars)

# #         # Update weights
# #         self.optimizer.apply_gradients(zip(gradients, trainable_vars))

# #         # Update the metrics configured in `compile()`.
# #         self.compiled_metrics.update_state(y, student_predictions)

# #         # Return a dict of performance
# #         results = {m.name: m.result() for m in self.metrics}
# #         results.update(
# #             {"student_loss": student_loss, "distillation_loss": distillation_loss}
# #         )
# #         return results

# #     def test_step(self, data):
# #         # Unpack the data
# #         x, y = data

# #         # Compute predictions
# #         y_prediction = self.student(x, training=False)

# #         # Calculate the loss
# #         student_loss = self.student_loss_fn(y, y_prediction)

# #         # Update the metrics.
# #         self.compiled_metrics.update_state(y, y_prediction)

# #         # Return a dict of performance
# #         results = {m.name: m.result() for m in self.metrics}
# #         results.update({"student_loss": student_loss})
# #         return results
    
# # n_classes = len(np.unique(pd.read_excel('Nexus5_RSSI_data.xlsx').pop('Label')))

# # train = pd.read_csv('nexus_student_train.csv')
# # test = pd.read_csv('nexus_test.csv')
# # train_exp = train.pop(str(train.shape[1]-1))
# # test_exp = test.pop(str(test.shape[1]-1))

# # n_inputs = train.shape[1]
# # h1_nodes = ceil(pow(2,log2(n_inputs+n_classes)))
# # student = tf.keras.models.Sequential()
# # student.add(tf.keras.Input(shape=(n_inputs,)))
# # student.add(tf.keras.layers.Dense(h1_nodes*4, activation='relu'))
# # student.add(tf.keras.layers.Dense(h1_nodes*2, activation='relu'))
# # student.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
# # student.summary()
# # student.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
# # # teacher = tf.keras.models.load_model('nexus_teacher')
# # # print(train.shape, train_exp.shape, test.shape, test_exp.shape)
# # # teacher.summary()

# # # # Initialize and compile distiller
# # # distiller = Distiller(student=student, teacher=teacher)
# # # distiller.compile(
# # #     optimizer=tf.keras.optimizers.Adam(),
# # #     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
# # #     student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
# # #     distillation_loss_fn=tf.keras.losses.KLDivergence(),
# # #     alpha=0.1,
# # #     temperature=10,
# # # )

# # # # Distill teacher to student
# # # distiller.fit(train, train_exp, epochs=100, validation_data=(test, test_exp))

# # # # Evaluate student on test dataset
# # # print('distillation result')
# # # distiller.evaluate(test, test_exp)

# # # print('teacher result')
# # # teacher.evaluate(test,test_exp)

# # # print('student result')
# # # student.evaluate(test,test_exp)
# # # student.save('student_test')

# # student.fit(train, train_exp, epochs=100, validation_data=(test, test_exp))
# # print('student result')
# # student.evaluate(test,test_exp)

# import tensorflow as tf
# import pandas as pd
# import numpy as np
# from math import ceil, log2
# from matplotlib import pyplot as plt
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.model_selection import KFold, train_test_split
# from sklearn.utils import class_weight, resample

# og = pd.read_excel('Nexus5_RSSI_data.xlsx').drop(columns=['X', 'Y'])
# og_exp = np.array(list(map(lambda x: x-1, og.pop('Label'))))
# train, test, train_exp, test_exp = train_test_split(og, og_exp, test_size=0.1, stratify=og_exp, random_state=1)
# n_classes = len(np.unique(og_exp))
# n_inputs = og.shape[1]

# train = MinMaxScaler().fit_transform(train.values)
# test = MinMaxScaler().fit_transform(test.values)

# n_members = 6 #5 for juindoorloc #6 for nexus
# # load models from file
# def load_all_models(n_models, model_name):
#     all_models = list()
#     for i in range(n_models):
#         filename = model_name + str(i)
#         # load model from file
#         model = tf.keras.models.load_model(filename)
#         # add to list of members
#         all_models.append(model)
#         print('>loaded %s' % filename)
#     return all_models

# # create stacked model input dataset as outputs from the ensemble
# def stacked_dataset(members, inputX):
#     stackX = None
#     for model in members:
#         yhat = model.predict(inputX, verbose=0)
#         # stack predictions into [rows, members, probabilities]
#         if stackX is None:
#             stackX = yhat
#         else:
#             stackX = np.dstack((stackX, yhat))
#     # flatten predictions to [rows, members x probabilities]
#     stackX = stackX.reshape(
#         (stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
#     return stackX

# def fit_stacked_model(members, inputX, inputy, valX, valy):
#     X = stacked_dataset(members, inputX)
#     Y = stacked_dataset(members, valX)
#     # model = tf.keras.models.load_model('nexus_sep_st_model_smaller')
#     # print('model reloaded for retraining')
#     h1_nodes = len(members)*n_classes
#     model = tf.keras.models.Sequential()
#     model.add(tf.keras.Input(shape=(h1_nodes,)))
#     model.add(tf.keras.layers.Dropout(0.2)) #0.2 for juindoorloc #0.1 for nexus 
#     model.add(tf.keras.layers.Dense(h1_nodes//256, activation='relu')) #/128 with 45 epochs acc 95.02
#     model.add(tf.keras.layers.Dropout(0.5))
#     model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
#     model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.summary()
#     EPOCHS = 240
#     model.fit(X, inputy, validation_data=(Y, valy), epochs=EPOCHS, verbose=2)
#     return model

# _, train, __, train_exp = train_test_split(train, train_exp, test_size=0.1, stratify=train_exp, random_state=1)
# train_exp = np.array(train_exp)

# members = load_all_models(n_members, 'nexus_kfold_ensemble')
# # fit stacked model on test dataset
# fit_stacked_model(members, train, train_exp, test, test_exp).save('nexus_sep_st_model_smaller')

