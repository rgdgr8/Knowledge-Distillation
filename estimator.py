import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from math import ceil
tf.get_logger().setLevel('INFO') 

train = pd.read_csv('/Users/rgdgr8/Documents/JUIndoorLoc-Training-data.csv').drop(columns=['Rs', 'Ts', 'Did', 'Hpr'])
test = pd.read_csv('/Users/rgdgr8/Documents/JUIndoorLoc-Test-data.csv').drop(columns=['Rs', 'Ts', 'Did', 'Hpr'])

train_y = train.pop('Cid')
test_y = test.pop('Cid')

data_to_class = {}
n_classes = 0
for i in train_y:
    if(i not in data_to_class):
        data_to_class[i] = n_classes
        n_classes += 1
train_y = train_y.apply(lambda x: data_to_class[x])
test_y = test_y.apply(lambda x: data_to_class[x])

my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

train = pd.DataFrame(StandardScaler().fit_transform(train.values), index=train.index, columns=train.columns)
test = pd.DataFrame(StandardScaler().fit_transform(test.values), index=test.index, columns=test.columns)

BATCH = 64
def input_fn(features, labels, training=True, batch_size=BATCH):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset = dataset.shuffle(BATCH).repeat()
    return dataset.batch(batch_size)

classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[(len(my_feature_columns)+n_classes)//2],
    n_classes=n_classes)

# train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(train, train_y, training=True), max_steps=1)
# eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(test, test_y, training=False))
# result = tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
# print(result)


for _ in range(40):
    classifier.train(input_fn=lambda: input_fn(train, train_y, training=True),steps=ceil(train.shape[0]/BATCH))
    eval_result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))
    print(eval_result)