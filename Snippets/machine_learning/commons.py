# -*- coding: utf-8 -*-
from time import strftime, gmtime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------- PREPROCESSING PRE-ML ----------------------------------------------
def standardize_features(train: pd.DataFrame, test: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Standardize all features except label to mean 0 variance 1.
    The test will be processed with the mean and the std of the training set.

    :param train: feature array of training set from where the std and the mean will be estimated
    :param test: feature array of the test set
    :return: tuple of the stand. train and test arrays
    """
    train_std = (train - train.mean()) / train.std()
    test_std = (test - train.mean()) / train.std()
    return train_std, test_std


def split_data(x, y, ratio, myseed=1):
    """
    x, y must be arrays
    Split the dataset based on the given ratio.
    Give train, labels, ratio (arrays: if data frame do df.values)
    Returns train, test, y_train, y_test arrays
    """
    # check input are arrays
    check_array(x)
    check_array(y)
    # set seed
    np.random.seed(myseed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te


# ---------------------------------------------- ML ----------------------------------------------
def clf_evaluation(clf, x_train, y_train, x_test, y_test):
    """ Evaluate the random forest given training and testing sets"""
    print('Evaluate classifier: train the model and predict')
    clf.fit(x_train, y_train)  # fit the model
    correctness: list = clf.predict(x_test) == y_test
    return sum(correctness)/len(correctness)  # accuracy


def my_cross_validation(X, y, clf, k_fold, ratio, seed):
    """ Evaluate the accuracy using cross-validating """
    pred_ratio = []
    # iterate through each train-test split
    print('Start cross validation {}'.format(strftime("%H:%M:%S", gmtime())))
    for k in range(k_fold):
        if k == k_fold/2:
            print('{} fold, 50% of CV done, {}'.format(k, strftime("%H:%M:%S", gmtime())))
        x_train, x_test, y_train, y_test = split_data(X, y, ratio, seed) # the k-th split
        accuracy = clf_evaluation(clf, x_train, y_train, x_test, y_test) # evaluate the result
        pred_ratio.append(accuracy)
    return np.mean(pred_ratio)


def check_array(array: np.array) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError('array must be an array not a {}'.format(type(array)))


def check_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a df not a {}'.format(type(df)))


def check_serie(serie: pd.Series) -> None:
    if not isinstance(serie, pd.Series):
        raise TypeError('serie must be a serie not a {}'.format(type(serie)))


def compute_metric(prediction: np.array, y: np.array, plot_the_confusion : bool = True) -> (list, dict):
    """ Compute metric f1, accuracy, precision and recall for the given labels and predictions
    :param prediction: np.array of the predicted labels
    :param y: np.array of true labels
    :return: tuple (confusion metric, {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1_score})
    """
    def confusion_Mx(prediction: np.array, y: np.array, plot_confusion: bool = True) -> list:
        """ Construct the confusing matrix. working only if labels are {0, 1}. Positive are 1, neg 0.
            TP: 1s in predictions that match 1s in y/1s in y
            FP: 1s in predictions that match 0s in y/0s in y
            TN: 0s in predictions that match 0s in y/0s in y
            FN: 0s in predictions that match 1s in y/1s in y
        """
        check_array(prediction)
        check_array(y)
        if len(set(prediction)) > 1:
            assert len(set(prediction)) == 2
            assert len(set(y)) == 2
            assert all([np.unique(y)[i] == np.unique(prediction)[i] for i in range(len(set(y)))])
        else:
            raise UserWarning('Predictions have only 1 class')

        # confusion matrix
        true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
        for true_lab, pred_lab in zip(y, prediction):
            # true positive: predicted positive and it’s true
            if true_lab == pred_lab and true_lab == 1:
                true_pos = true_pos + 1
            # true negative: predicted negative and it’s true
            if true_lab == pred_lab and true_lab == 0:
                true_neg = true_neg + 1
            # false positive: predicted positive and it’s false
            if true_lab != pred_lab and pred_lab == 1 and true_lab == 0:
                false_pos = false_pos + 1
            # predicted negative and it’s false
            if true_lab != pred_lab and pred_lab == 0 and true_lab == 1:
                false_neg = false_neg + 1

        #  shape = nb of elements under the condition
        TP = true_pos / np.where(y == 1)[0].shape[0]
        TN = true_neg / np.where(y == 0)[0].shape[0]
        FP = false_pos / np.where(y == 0)[0].shape[0]
        FN = false_neg / np.where(y == 1)[0].shape[0]

        mx: list = [[TP, FP], [FN, TN]]
        # plot the Matrix
        if plot_confusion:
            fig, ax = plt.subplots(figsize=(9, 9))
            ax.imshow(mx, cmap=plt.cm.Blues)
            ax.text(0, 0, 'True positive: {:2f}'.format(mx[0][0]), ha="center", va="center", color="w", size=14)
            ax.text(1, 0, 'False positive: {:2f}'.format(mx[0][1]), ha="center", va="center", color="b", size=14)
            ax.text(0, 1, 'False negative: {:2f}'.format(mx[1][0]), ha="center", va="center", color="b", size=14)
            ax.text(1, 1, 'True negative: {:2f}'.format(mx[1][1]), ha="center", va="center", color="w", size=14)
            plt.title('Confusion Matrix', size=14)
            ax.axis('off')

            fig.tight_layout()
            plt.show()
        return mx

    def calculate_values(confusion_Mx):
        """ Calculate accuracy, precision, recall, and F1 - score with respect to the positive and the negative class
        """
        TP = confusion_Mx[0][0]
        FP = confusion_Mx[0][1]
        FN = confusion_Mx[1][0]
        TN = confusion_Mx[1][1]
        accuracy = (TP + TN) / (TP + FP + FN + TN)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * precision * recall/ (precision + recall)
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1_score}

    check_array(prediction)
    check_array(y)
    confusion_mx = confusion_Mx(prediction, y, plot_confusion=plot_the_confusion)
    metrics = calculate_values(confusion_mx)
    return confusion_mx, metrics
