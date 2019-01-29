import seaborn as sns
import matplotlib.pylab as plt
from sklearn.ensemble import RandomForestClassifier
from machine_learning.commons import *
# from commons import *
# from pythonsnippets.machine_learning.commons import *


def grid_search_RF(X, y, k_fold, num_depth, num_trees, plot_heatmap=True, rand_seed=1, tt_ratio=0.5):
    """ Grid search using CV with random forest. X, y must be arrays """
    # check input are arrays
    check_array(X)
    check_array(y)
    accuracy = np.zeros(len(num_depth) * len(num_trees))  # accuracy array

    ct = 0  # counter
    print('Start grid search')
    for n_depth in num_depth:
        for n_trees in num_trees:
            print('\n Try param {} and {}, {}'.format(n_trees, n_depth, strftime("%H:%M:%S", gmtime())))
            # for each grid, define a random forest and evaluate the forest using crossvalidation
            RForest = RandomForestClassifier(n_estimators=n_trees, max_depth=n_depth)  # change here with other clf
            accuracy[ct] = my_cross_validation(X, y, RForest, k_fold, tt_ratio, rand_seed)
            ct += 1

    accuracy = accuracy.reshape(len(num_depth), len(num_trees))
    ind = np.unravel_index(np.argmax(accuracy, axis=None), accuracy.shape)
    best_hyperparameters = {
        'max_accuracy': accuracy[ind],
        'best_depth': num_depth[ind[0]],
        'best_tree': num_trees[ind[1]]
    }
    print("Best hyperparameters are: best_depth: {} n_estimators (trees) : {}".format(
        num_depth[ind[0]],
        num_trees[ind[1]])
    )
    if plot_heatmap:
        plot_results_RF(accuracy, num_depth, num_trees)
    return best_hyperparameters


def grid_search_RF_noCV(X, y, num_depth, num_trees, plot_heatmap=True):
    """ Grid search using CV with random forest. X, y must be arrays """
    # check input are arrays
    check_array(X)
    check_array(y)
    accuracy = np.zeros(len(num_depth) * len(num_trees))  # accuracy array

    ct = 0  # counter
    print('Start grid search')
    for n_depth in num_depth:
        for n_trees in num_trees:
            print('\n Try param {} and {}, {}'.format(n_trees, n_depth, strftime("%H:%M:%S", gmtime())))
            # for each grid, define a random forest and evaluate the forest using crossvalidation
            RForest = RandomForestClassifier(n_estimators=n_trees, max_depth=n_depth)  # change here with other clf
            accuracy[ct] = clf_evaluation(RForest, X, y, X, y)
            ct += 1

    accuracy = accuracy.reshape(len(num_depth), len(num_trees))
    ind = np.unravel_index(np.argmax(accuracy, axis=None), accuracy.shape)
    best_hyperparameters = {
        'max_accuracy': accuracy[ind],
        'best_depth': num_depth[ind[0]],
        'best_tree': num_trees[ind[1]]
    }
    print("\n Best hyperparameters are: best_depth: {} n_estimators (trees) : {}".format(
        num_depth[ind[0]], num_trees[ind[1]])
    )
    if plot_heatmap:
        plot_results_RF(accuracy, num_depth, num_trees)
    return best_hyperparameters


def plot_results_RF(accuracy, grid_params1, grid_params2) -> None:
    """ Plot heat map of the results of the grid search"""
    # plot the result of grid search
    plt.figure(figsize=(8, 6))
    ax1 = sns.heatmap(
        accuracy,
        vmin=np.min(accuracy),  # vmin = 0.8,
        vmax=np.max(accuracy),  # vmax = 0.86
        cmap="YlGnBu",
        xticklabels=grid_params2,
        yticklabels=grid_params1
    )

    plt.title('accuracy distribution', fontsize=14)
    plt.xlabel('the number of trees', fontsize=14)
    plt.ylabel('depth', fontsize=14)
    plt.show()

    print('cross-validation accuracy of {:.2f}'.format(accuracy.max()*100))


def plot_features_importance(clf: RandomForestClassifier, X: pd.DataFrame) -> None:
    """ Plot feature importance per Random Forest Classifier.
    REMEMBER: retrain with best hp found by grid search CV.
    E.g.:
        # retrain
        clf = RandomForestClassifier(n_estimators=50, max_depth=10)  # found by gridsearchCV
        clf.fit(X_train, y_train)

        # plot
        plot_features_importance(clf, features)

    :param clf: RandomForestClassifier fitted
    :param X: X is the train data, with all columns used to fit the RF.
    :return: None
    """
    check_dataframe(X)
    forest_feature_imp = pd.Series(
        clf.feature_importances_,
        index=X.columns)\
        .sort_values(ascending=False)

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    sns.barplot(x=forest_feature_imp, y=forest_feature_imp.index, ax=ax)
    # Add labels
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important features with {} score".format(clf.criterion))
    plt.show()
