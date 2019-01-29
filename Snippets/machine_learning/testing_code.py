from commons import *
from grid_search_RF import *
from grid_search_AdaBoostClassifier import *

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import stats
from sklearn.datasets import make_classification
import numpy as np

def main():


    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    #grid_search_AdaBoostClassifier(linearly_separable[0], linearly_separable[1], k_fold=3,
                                  # n_estimators=[1, 5, 10, 15, 20, 25, 30, 35, 40, 50],
                                  # learning_rate=[0.1, 0.2, 0.3, 0.5, 1, 3, 4, 5, 10])

    grid_search_RF(linearly_separable[0], linearly_separable[1], k_fold=3, num_depth=[1,5,10,15,20,30,40], num_trees = [2,5,10,20,30,40,50])


if __name__ == '__main__':
    main()