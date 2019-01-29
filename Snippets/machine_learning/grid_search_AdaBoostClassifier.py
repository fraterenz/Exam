from time import strftime, gmtime

import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.ensemble import AdaBoostClassifier
#from machine_learning.commons import *
from machine_learning.commons import check_array, my_cross_validation


def grid_search_AdaBoostClassifier(X, y, k_fold, n_estimators, learning_rate, plot_heatmap=True, rand_seed=1, tt_ratio=0.5):
    """ Grid search using CV with random forest. X, y must be arrays """
    # check input are arrays
    check_array(X)
    check_array(y)
    accuracy = np.zeros(len(n_estimators) * len(learning_rate))  # accuracy array

    ct = 0  # counter
    print('Start grid search')
    for n_est in n_estimators:
        for n_rate in learning_rate:
            print('\n Try param {} and {}, {}'.format(n_rate, n_est, strftime("%H:%M:%S", gmtime())))
            # for each grid, define a random forest and evaluate the forest using crossvalidation
            AdaBoost_ = AdaBoostClassifier(n_estimators=n_est, learning_rate=n_rate)
            accuracy[ct] = my_cross_validation(X, y, AdaBoost_, k_fold, tt_ratio, rand_seed)
            ct += 1

    accuracy = accuracy.reshape(len(n_estimators), len(learning_rate))
    ind = np.unravel_index(np.argmax(accuracy, axis=None), accuracy.shape)
    best_hyperparameters = {
        'max_accuracy': accuracy[ind],
        'best_estimators': n_estimators[ind[0]],
        'best_learning_rate': learning_rate[ind[1]]
    }
    print("\n Best hyperparameters are: best_estimators: {} best_learning_rate: {}".format(
        n_estimators[ind[0]],
        learning_rate[ind[1]])
    )
    if plot_heatmap:
        plot_results_AdaBoostClassifier(accuracy, n_estimators, learning_rate)
    return best_hyperparameters


def plot_results_AdaBoostClassifier(accuracy, grid_params1, grid_params2):
    """ Plot heat map of the results of the grid search"""
    # plot the result of grid search
    plt.figure(figsize=(8, 6))
    ax1 = sns.heatmap(accuracy, vmin=np.min(accuracy), vmax=np.max(accuracy), cmap="YlGnBu", xticklabels=grid_params2,
                      yticklabels=grid_params1)
    plt.title('accuracy distribution', fontsize=14)
    plt.xlabel('the number of estimators', fontsize=14)
    plt.ylabel('learning_rate', fontsize=14)
    plt.show()

    print('cross-validation accuracy of {:.2f}'.format(accuracy.max()*100))
