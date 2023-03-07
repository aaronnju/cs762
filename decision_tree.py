
# https://www.section.io/engineering-education/entropy-information-gain-machine-learning/

from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree
from matplotlib import pyplot as plt

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