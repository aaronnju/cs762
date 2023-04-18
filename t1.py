import numpy as np
from sklearn import tree
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import entropy
import graphviz
import pydotplus

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


def entropy(labels):
     value, counts = np.unique(labels, return_counts=True)
     norm_counts = counts / counts.sum()
     return -(norm_counts * np.log2(norm_counts)).sum()


# https://www.featureranking.com/tutorials/machine-learning-tutorials/information-gain-computation/
def giniIndices(labels):
     value, counts = np.unique(labels, return_counts=True)
     norm_counts = counts / counts.sum()
     gini_index = 1 - np.sum(np.square(norm_counts))


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

    dot_data = tree.export_graphviz(clf,
                                    # feature_names=X.columns,
                                    # class_names=X.columns,
                                    filled=True)
    # graph = graphviz.Source(dot_data)
    # png_bytes = graph.pipe(format='png')
    # with open('./dtree_pipe.png', 'wb') as f:
    #     f.write(png_bytes)

    # graph = graphviz.Source(dot_data, format="png")
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('tree.png')

    # fig, ax = plt.subplots(figsize=(6, 6))
    # tree.plot_tree(clf, ax=ax, feature_names=['Colour', 'Length', 'Size', 'Brightness', 'Shape', 'Class'])
    # plt.show()


question3()

