import matplotlib.pyplot as plt

# accuracies = [82.10, 81.80, 82.74, 84.04, 79.73, 81.00, 79.97, 65.31, 81.10]
# models = list(map(str,range(1,len(accuracies)+1)))
accuracies = [81.10, 95.54, 93.49]
dataset = ['DataSet1', 'DataSet2', 'DataSet3']

plt.bar(dataset, accuracies, width=0.4)
plt.xlabel("DATASETS")
plt.ylabel("Max Accuracies with base neural network classifiers")
plt.show()