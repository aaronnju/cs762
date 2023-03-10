# referencing https://www.section.io/engineering-education/entropy-information-gain-machine-learning/
# before using this code, you should
# pip install sklearn matplotlib numpy

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn import tree
from matplotlib import pyplot as plt


def entropy(labels):
     value, counts = np.unique(labels, return_counts=True)
     norm_counts = counts / counts.sum()
     return -(norm_counts * np.log2(norm_counts)).sum()


# https://www.featureranking.com/tutorials/machine-learning-tutorials/information-gain-computation/
def giniIndices(labels):
     value, counts = np.unique(labels, return_counts=True)
     norm_counts = counts / counts.sum()
     gini_index = 1 - np.sum(np.square(norm_counts))


print(entropy(['no', 'yes', 'yes', 'no']))

# using online data
# iris = load_iris()
# X = iris.data
# y = iris.target

# or data from tutorial
# 'A', 'B', 'B1', 'Y'
M = np.array([
     [0, 0, 1, 0],
     [1, 0, 1, 1],
     [1, 0, 1, 1],
     [1, 1, 0, 0]])
X = M[:, 0:3]
y = M[:, -1]

print(X)
print(y)

# build decision tree
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=1)
# max_depth represents max level allowed in each tree, min_samples_leaf minumum samples storable in leaf node

# fit the tree to iris dataset
clf.fit(X, y)

# plot decision tree
fig, ax = plt.subplots(figsize=(6, 6)) # figsize value changes the size of plot
# tree.plot_tree(clf, ax=ax, feature_names=['sepal length','sepal width','petal length','petal width'])
tree.plot_tree(clf, ax=ax, feature_names=['A', 'B', 'B1', 'Y'])
plt.show()