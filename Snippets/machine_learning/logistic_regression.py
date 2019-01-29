import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from machine_learning.commons import compute_metric, check_array


def predict_test_logistic(Xtr: np.array, ytr: np.array, Xte: np.array, yte: np.array, threshold) -> (np.array, LogisticRegression, np.float):
    """ Fit logistic, compute probabilities and binary predictions with accuracy.
    The predictions are positive with label equals to 1 or negative with 0.
    :param Xtr:
    :param ytr:
    :param Xte:
    :param yte:
    :param threshold:
    :return:
    """
    # check that they are not pandas
    check_array(Xtr)
    check_array(ytr)
    check_array(Xte)
    check_array(yte)
    # declare classifier
    logistic = LogisticRegression(solver='lbfgs')

    # fit the model
    logistic.fit(Xtr, ytr)

    # compute probabilities
    proba: np.array = logistic.predict_proba(Xte)

    # Compute binary predictions from probabilities with threshold
    predictions = np.fromiter(map(lambda x: 1 if x[0] > threshold else 0, proba), dtype=np.uint8)

    # compute accuracy, 1 is positive 0 is negative class
    correct: list = predictions == yte
    accuracy = sum(correct)/len(correct)
    print('Accuracy {} with {} threshold'.format(accuracy, threshold))
    return predictions, logistic, accuracy


def evolution_metric_with_threshold(proba, y_test, plot_curves: bool = True) -> dict:
    """ Calculate the evolution of metrics depending on the threshold. Proba are the probabilities predicted
    by logistic regression, i.e. proba: np.array = logistic.predict_proba(Xtest)
    """
    def plot_metrics_threshold(the_accuracies, the_precisions, the_recalls, the_f1_scores) -> None:
        """ Plot the evolution of metrics depending on the threshold. Plot metrics for each threshold"""
        fig, ax1 = plt.subplots(1, 1, figsize=(11, 9))
        ax1.plot(the_accuracies, color='r')
        ax1.plot(the_precisions, color='y')
        ax1.plot(the_recalls, color='b')
        ax1.plot(the_f1_scores, color='g')
        ax1.legend(['accuracy', 'precision', 'recall', 'f1 score'])
        ax1.set_title('With respect to positive classe')
        ax1.set_xlabel('threshold (%)')
        plt.show()

    accuracies, precisions, recalls, f1_scores = [], [], [], []
    # for each threshold going from 0 to 1, so 0% to 100%
    for threshold in range(99, 0, -1):
        # transform proba into binary predictions according to the threshold
        predictions = np.fromiter(map(lambda x: 0 if x[0] > threshold/100 else 1, proba), dtype=np.uint8)
        # compute confusion matrix and metrics like f1, accuracy, recall, precision
        matrix, metrics = compute_metric(predictions, y_test, plot_the_confusion=False)
        a, p, r, f = metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']
        accuracies.append(a)
        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f)
        # store metrics
        metrics_per_threshold = {'accuracies': accuracies, 'precisions': precisions, 'recalls': recalls,
                                 'f1_score': f1_scores}
    if plot_curves:
        plot_metrics_threshold(
            the_accuracies=metrics_per_threshold['accuracies'],
            the_recalls=metrics_per_threshold['recalls'],
            the_precisions=metrics_per_threshold['precisions'],
            the_f1_scores=metrics_per_threshold['f1_score']
        )
    return metrics_per_threshold
