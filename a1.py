from sklearn.datasets import load_iris
from scipy.stats import entropy
from scipy import stats
import pandas as pd
import numpy as np
from sklearn import tree
from matplotlib import pyplot as plt
from statistics import median
from statistics import mean
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
from mlxtend.evaluate import paired_ttest_5x2cv


def fill_missing_feature(column_name, column_data, log):
    decimal_items = list(filter(lambda i: (isinstance(i, int) or not np.isnan(i)), column_data))
    avg = mean(decimal_items)
    if log:
        print(f"Replace {column_name}'s missing value with median={avg}. {len(decimal_items)}/{len(column_data)}")
    for index, value in column_data.items():
        if np.isnan(value):
            column_data.at[index] = avg
    return column_data


def fill_missing_values(dataset, log):
    for (column_name, column_data) in dataset.items():
        num_column_data = pd.to_numeric(column_data, errors='coerce')
        if num_column_data.hasnans:
            fill_missing_feature(column_name, num_column_data, log)
        dataset[column_name] = num_column_data


def get_accuracy(x_test, y_test, estimator):
    return accuracy_score(y_test, estimator.predict(x_test))


def calc_accuracy(X, y, estimator):
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)
    estimator.fit(x_train, y_train)
    predictions = estimator.predict(x_test)
    # print(predictions[:5])
    print(accuracy_score(y_test, predictions))
    fig, ax = plt.subplots(figsize=(6, 6))
    tree.plot_tree(estimator, ax=ax, feature_names=X.columns, filled=True)
    plt.show()


def plot_tree(title, estimator, feature_names, fontsize=4):
    import graphviz
    fig, ax = plt.subplots(figsize=(16, 10))
    tree.plot_tree(estimator, ax=ax, feature_names=feature_names, filled=True, fontsize=fontsize)
    ax.title.set_text(title)


def make_stump_tree(file, x_train, x_test, y_train, y_test):
    estimator = tree.DecisionTreeClassifier(criterion='entropy', max_depth=1, random_state=42)
    estimator.fit(x_train, y_train)
    print(f"Make a stump tree with score={get_accuracy(x_test, y_test, estimator)} max_depth={estimator.tree_.max_depth}")
    plot_tree(f'Stump tree({file})', estimator, x_train.columns, 20)
    return {'name': 'Stump tree', 'estimator': estimator}


def make_unpruned_tree(file, x_train, x_test, y_train, y_test):
    estimator = tree.DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=42)
    estimator.fit(x_train, y_train)
    print(f"Make a unpruned tree with score={get_accuracy(x_test, y_test, estimator)} max_depth={estimator.tree_.max_depth}")
    plot_tree(f'Unpruned tree({file})', estimator, x_train.columns)
    return {'name': 'Unpruned tree', 'estimator': estimator}


def select_hyperparameters(file, x_train, x_test, y_train, y_test):
    from sklearn.model_selection import GridSearchCV
    params = {
        # 'criterion': ['gini', 'entropy'],
        'max_depth': [None, 2, 4, 6, 8, 10],
        'min_samples_split': [2, 3, 4, 5, 6],
        'min_samples_leaf': [2, 3, 4, 5, 6],
        # 'max_features': [None, 'sqrt', 'log2', 0.2, 0.4, 0.6, 0.8],
        # 'splitter': ['best', 'random']
    }
    # fold_map = {'arrhythmia.csv': 2, 'BCP.csv': 5, 'website-phishing.csv': 5}
    gscv = GridSearchCV(
        estimator=tree.DecisionTreeClassifier(random_state=42),
        param_grid=params,
        # cv=fold_map[file],
        n_jobs=1,
        verbose=1,
    )
    # print(f'y_train distinct value count: {len(np.unique(y_train))}')
    # print(f'y_train distinct value count: {np.unique(y_train)}')
    gscv.fit(x_train, y_train)
    print(f"select hyper-parameters tree best param: {gscv.best_params_}")
    print(f"select hyper-parameters with score={get_accuracy(x_test, y_test, gscv.best_estimator_)} max_depth={gscv.best_estimator_.tree_.max_depth}")
    plot_tree(f' select hyper-parameters best_estimator({file})', gscv.best_estimator_, x_train.columns)
    return gscv.best_estimator_


# {'max_depth': 6, 'min_samples_leaf': 5, 'min_samples_split': 4}
def make_prepruned_tree(file, x_train, x_test, y_train, y_test, hyper_parameters):
    estimator = tree.DecisionTreeClassifier(criterion='entropy', max_depth=hyper_parameters[0],
                                            min_samples_split=hyper_parameters[1], min_samples_leaf=hyper_parameters[2], random_state=42)
    estimator.fit(x_train, y_train)
    print(
        f"Make a pre-pruned tree with score={get_accuracy(x_test, y_test, estimator)} max_depth={estimator.tree_.max_depth}")
    plot_tree(f'Pre-pruned tree({file})', estimator, x_train.columns, 14)
    return {'name': 'Pre-pruned tree', 'estimator': estimator}


def distribution(data):
    sns.pairplot(data=data, hue=data.columns[-1])
    plt.show()


def transform_string():
    from sklearn.compose import make_column_transformer
    from sklearn.preprocessing import OneHotEncoder
    column_transformer = make_column_transformer((OneHotEncoder(), ['Sex', 'Embarked']), remainder='passthrough')
    # x_train = column_transformer.fit_transform(x_train)


def calc_p(X, y):
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    print(est2.summary())


# https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
def analyze_post_pruned_tree(file, x_train, y_train, x_test, y_test):
    clf = tree.DecisionTreeClassifier(random_state=42)
    path = clf.cost_complexity_pruning_path(x_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    fig, axs = plt.subplots(2, 2)
    fig.suptitle(file)
    axs[0, 0].plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
    axs[0, 0].set_xlabel("effective alpha")
    axs[0, 0].set_ylabel("total impurity of leaves")
    axs[0, 0].set_title("Total Impurity vs effective alpha for training set")

    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = tree.DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
        clf.fit(x_train, y_train)
        clfs.append(clf)
    print(f"Number of nodes in the last tree is: {clfs[-1].tree_.node_count} with ccp_alpha: {ccp_alphas[-1]}")

    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]

    node_counts = [clf.tree_.node_count for clf in clfs]
    depth = [clf.tree_.max_depth for clf in clfs]
    axs[0, 1].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
    axs[0, 1].set_xlabel("alpha")
    axs[0, 1].set_ylabel("number of nodes")
    axs[0, 1].set_title("Number of nodes vs alpha")
    axs[1, 0].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
    axs[1, 0].set_xlabel("alpha")
    axs[1, 0].set_ylabel("depth of tree")
    axs[1, 0].set_title("Depth vs alpha")
    fig.tight_layout()

    train_scores = [clf.score(x_train, y_train) for clf in clfs]
    test_scores = [clf.score(x_test, y_test) for clf in clfs]

    axs[1, 1].set_xlabel("alpha")
    axs[1, 1].set_ylabel("accuracy")
    axs[1, 1].set_title("Accuracy vs alpha for training and testing sets")
    axs[1, 1].plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    axs[1, 1].plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    axs[1, 1].legend()


def make_post_pruned_tree(file, x_train, y_train, x_test, y_test, ccp_alpha):
    estimator = tree.DecisionTreeClassifier(criterion='entropy', ccp_alpha=ccp_alpha, random_state=42)
    estimator.fit(x_train, y_train)
    print(f"Make a post pruned tree with score={get_accuracy(x_test, y_test, estimator)} max_depth={estimator.tree_.max_depth}")
    plot_tree(f'{file} Post-pruned tree', estimator, x_train.columns)
    return {'name': 'Post-pruned tree', 'estimator': estimator}


# https://towardsdatascience.com/paired-t-test-to-evaluate-machine-learning-classifiers-1f395a6c93fa
def compare(a, b, X, y, x_test, y_test):
    print(f'compare  {a["name"]} VS {b["name"]}')
    s1 = a["estimator"].score(x_test, y_test)
    s2 = b["estimator"].score(x_test, y_test)
    print(f'\tModel {a["name"]} accuracy: %.2f%%' % (s1 * 100))
    print(f'\tModel {b["name"]} accuracy: %.2f%%' % (s2 * 100))
    t, p = paired_ttest_5x2cv(estimator1=a["estimator"], estimator2=b["estimator"], X=X, y=y)
    alpha = 0.05
    print('\tt statistic: %.3f' % t)
    print('\taplha ', alpha)
    print('\tp value: %.3f' % p)

    if p > alpha:
        print("\tFail to reject null hypotesis")
    else:
        print("\tReject null hypotesis")


def split_data(df):
    X = df.copy()
    y = X.pop(df.columns[-1])
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # random_state=42
    return X, y, x_train, x_test, y_train, y_test


# scipy.stats.ttest_ind
# TODO website-phishing ccp_alpha is abnormal
def main():
    # {'max_depth': 6, 'min_samples_leaf': 5, 'min_samples_split': 4}
    hper_parameters = {'arrhythmia.csv': [6, 5, 4], 'BCP.csv': [6, 6, 2], 'website-phishing.csv': [21, 2, 5]}
    ccp_alpha = {'arrhythmia.csv': 0.01256, 'BCP.csv': 0.003, 'website-phishing.csv': 0.003}
    for csv in ['arrhythmia.csv', 'BCP.csv', 'website-phishing.csv']:
    # for csv in ['arrhythmia.csv']:
    # for csv in ['BCP.csv']:
    # for csv in ['website-phishing.csv']:
        print('Processing:' + csv)
        df = pd.read_csv("/Users/shuming/Source/UoA/cs762/data/" + csv)
        # Question 1
        # Answer:
        fill_missing_values(df, True)

        # Split data for all subsequent jobs.
        X, y, x_train, x_test, y_train, y_test = split_data(df)

        # Question 2
        # Answer:
        # FIXME a decision stump: Only One tree is enough?
        # stump_tree = make_stump_tree(csv, x_train, x_test, y_train, y_test)
        # unpruned_tree = make_unpruned_tree(csv, x_train, x_test, y_train, y_test)
        analyze_post_pruned_tree(csv, x_train, y_train, x_test, y_test)
        # post_pruned_tree = make_post_pruned_tree(csv, x_train, y_train, x_test, y_test, ccp_alpha[csv])

        # Question 3
        # Answer:
        # FIXME some of them have no different
        # processing:website-phishing.csv
        # Make a stump tree with score=0.8821223997588182 max_depth=1
        # Make a unpruned tree with score=0.964124208622249 max_depth=23
        # Make a post pruned tree with score=0.929454326198372 max_depth=7
        # select hyper-parameters with score=0.9427193246909858 max_depth=18
        # FIXME the hyper-parameters differ each time, and max_depth is None
        # select hyper-parameters tree best param: {'max_depth': None, 'min_samples_leaf': 3, 'min_samples_split': 3}
        # select hyper-parameters tree best param: {'max_depth': None, 'min_samples_leaf': 3, 'min_samples_split': 4}
        # FIXME for some dataset sample is not enough to 5-fold
        # select_hyperparameters(csv, x_train, x_test, y_train, y_test)

        # Question 5
        # Answer:
        # pre_pruned_tree[csv] = make_prepruned_tree(csv, x_train, x_test, y_train, y_test, hper_parameters[csv])

        # Question 4
        # Answer:
        # compare(stump_tree, unpruned_tree, X, y, x_test, y_test)
        # compare(stump_tree, post_pruned_tree, X, y, x_test, y_test)
        # compare(post_pruned_tree, unpruned_tree, X, y, x_test, y_test)

        plt.show()

main()
