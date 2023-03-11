import numpy as np
from sklearn import tree
from matplotlib import pyplot as plt
import pandas as pd


def question2():
    # 'A', 'B', 'C', 'Y'
    M = np.array([
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 1]])

    X = M[:, 0:3]
    y = M[:, -1]

    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=1)
    clf.fit(X, y)

    fig, ax = plt.subplots(figsize=(6, 6))
    tree.plot_tree(clf, ax=ax, feature_names=['A', 'B', 'C', 'Y'])
    plt.show()


def question3():

    data = pd.DataFrame()
    data['Colour'] = ['red','red','red','red', 'red', 'blue']
    data['Length'] = ['long','long','long','short','short','short']
    data['Size'] = ['larger', 'small', 'small', 'larger', 'larger', 'larger']
    data['Brightness'] = ['bright','bright','bright','dull','bright','bright']
    data['Shape'] = ['triangle', 'circle', 'triangle', 'circle', 'triangle', 'triangle']
    data['Class'] = ['TRUE', 'FALSE', 'TRUE', 'FALSE', 'TRUE', 'FALSE']
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=6, min_samples_leaf=1)
    print(data[['Colour', 'Length', 'Size', 'Brightness', 'Shape', 'Class']])
    one_hot_data = pd.get_dummies(data[['Colour', 'Length', 'Size', 'Brightness', 'Shape']], drop_first=True)
    print(one_hot_data)
    clf.fit(one_hot_data, data['Class'])

    fig, ax = plt.subplots(figsize=(6, 6))
    tree.plot_tree(clf, ax=ax, feature_names=['Colour', 'Length', 'Size', 'Brightness', 'Shape', 'Class'])
    plt.show()


question3()

