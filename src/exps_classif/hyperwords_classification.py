#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn import neural_network
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection._split import train_test_split


def get_best_svm_params(X, Y):
    C_vals = [] # 11 values [0.03125, 0.125, 0.5, 2.0, 8.0, 32.0, 128.0, 512.0, 2048.0, 8192.0, 32768.0]
    for i in range(-5, 16, 2):
        C_vals.append(2.0**i)
    gamma_vals = [] # 10 values [3.05e-05, 0.00012, 0.00048, 0.00195, 0.00781, 0.03125, 0.125, 0.5, 2.0, 8.0]
    for i in range(-15, 4, 2):
        gamma_vals.append(2.0**i)
    params = {'C': C_vals, 'gamma': gamma_vals}
    classif_model = None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
#         classif_model = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=params,
#                                      cv=5, iid=False, n_jobs=-1, scoring='f1_macro', error_score=0.0)
        classif_model = RandomizedSearchCV(SVC(kernel='rbf'),
                                      param_distributions=params,
                                      n_iter = 20, n_jobs = -1, cv = 3,
                                      scoring = 'f1_macro', iid=False, error_score=0.0)
    classif_model.fit(X, Y)
    return classif_model.best_params_


def main(argv = None):
    if argv is None:
        argv = sys.argv
        
    dataset_filename = argv[1]
    bohw_filename = argv[2]
    num_folds = int(argv[3])
    shuffle_opt = bool(argv[4])
    scores_filename = argv[5]
    classif_type = argv[6]
    metric = argv[7]    # f1_macro
    
    print("Reading classes... ", end="", flush=True)
    Y = np.ravel(pd.read_csv(dataset_filename, sep=';', header=None,
                                   names=["id", "class", "text"],
                                   index_col=0)["class"].values)
    
    print("Reading data... ", end="", flush=True)
    X = pd.read_csv(bohw_filename, sep=',', header=0, index_col=0).values
    print("OK!")
    
    print("Training classifier:")
    classif = None
    if classif_type == "knn":
        classif = KNeighborsClassifier(1)
    elif classif_type == "nb":
        classif = GaussianNB(var_smoothing=1e-9)
    elif classif_type == "lr":
        classif = LogisticRegression(max_iter=100000000)
    elif classif_type == "rf":
        classif = RandomForestClassifier(n_estimators=100, max_depth=120)
    elif classif_type == "mlp":
        classif = neural_network.MLPClassifier(hidden_layer_sizes=(100,))
    elif classif_type == "svm":
        X, tuning_X, Y, tuning_Y = train_test_split(X, Y, test_size=0.1)
        best_params = get_best_svm_params(tuning_X, tuning_Y)
        classif = SVC(kernel='rbf', C = best_params['C'], gamma = best_params['gamma'],
                      decision_function_shape="ovr")
    cv = StratifiedKFold(n_splits=num_folds, shuffle=shuffle_opt)
    scores = pd.DataFrame(  pd.Series(cross_val_score(classif, X, Y, cv=cv, scoring=metric)),
                            columns=["scores"])
    scores.index.name = "fold"
    print("OK!")
    
    print("Saving scores... ", end="", flush=True)
    scores.to_csv(scores_filename, sep=',', float_format="%.4f")
    print("OK", flush=True)


if __name__ == '__main__':
    main()
