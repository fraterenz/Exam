import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from machine_learning.commons import my_cross_validation, check_dataframe, check_serie


def find_feature(X: pd.DataFrame, y: pd.Series, clf, k_fold=5, ratio=0.6, seed=42):
    """ For a feature matrix, find the least important feature and the accuracy after removing it.
    """
    check_dataframe(X)
    check_serie(y)

    accuracies = []
    num_features = X.shape[1]

    # interate columns for elimination
    for i in range(num_features):
        current_X: np.array = X.drop(X.columns[i], axis=1).values  # X without the i-th column
        accuracy = my_cross_validation(X=current_X, y=y.values, clf=clf, k_fold=k_fold, ratio=ratio, seed=seed)
        accuracies.append(accuracy)

    col2remove = accuracies.index(max(accuracies))
    # return the index of the least important feature and the highest accuracy
    return col2remove, max(accuracies)


def greedy_backward_selection(X: pd.DataFrame, y: pd.Series, clf, ks: list, k_fold=5, ratio=0.6, seed=42):
    """ Backward greedy selection of the most relevant features using accuracy and cross validation.
    :param X: training dataset in DataFrame format
    :param y: training labels, DataFrame
    :param clf: classifier you want to use
    :param ks: list of numbers of more features to obtain after selection [1, 5, 10]
    :param k_fold: for the cross valdiation in the process of feature selection
    :param ratio: training testing of the corss validation
    :param seed: seed
    :return: accura
    """
    check_dataframe(X)
    check_serie(y)

    greedy_X: pd.DataFrame = X.copy()
    num_features = X.shape[1]
    acc: list = []
    cols_removed: list = []

    # iteratively remove features until one left
    for i in range(num_features - 1):
        print(f"We are finding the {i}th feature.", end='\r')
        # find the least important feature and drop it
        col_index, max_acc = find_feature(greedy_X, y, clf, k_fold=k_fold, ratio=ratio, seed=seed)
        # keep track of the col removed
        cols_removed.append(X.columns[col_index])
        # remove the column
        greedy_X = greedy_X.drop(greedy_X.columns[col_index], axis=1)
        num_f = num_features - i - 1  # number of features left

        # store the performance when there are k features left
        if num_f in ks:
            acc.append(max_acc)

    return acc, greedy_X, cols_removed


def plot_backwards(ks: list, acc: list) -> None:
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.title('Accuracy of the prediction')
    plt.plot(ks, acc)
    plt.show()
