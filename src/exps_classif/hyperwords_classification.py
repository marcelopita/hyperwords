#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import linear_model


def main(argv = None):
    if argv is None:
        argv = sys.argv
        
    dataset_filename = argv[1]
    bohw_filename = argv[2]
    num_folds = int(argv[3])
    shuffle_opt = bool(argv[4])
    scores_filename = argv[5]
    metric = argv[6]    # f1_macro
    
    print("Reading targets... ", end="", flush=True)
    targets = np.ravel(pd.read_csv(dataset_filename, sep=';', header=None,
                                   names=["id", "class", "text"],
                                   index_col=0)["class"].values)
    print("OK!")
    
    print("Reading data... ", end="", flush=True)
    data = pd.read_csv(bohw_filename, sep=',', header=0, index_col=0).values
    print("OK!")
    
    print("Training classifier:")
    classif = linear_model.LogisticRegression(max_iter=1000000)
    cv = StratifiedKFold(n_splits=num_folds, shuffle=shuffle_opt)
    scores = pd.DataFrame(  pd.Series(cross_val_score(classif, data,
                                                      targets,
                                                      cv=cv, scoring=metric)),
                            columns=["scores"])
    scores.index.name = "fold"
    print("OK!")
    
    print("Saving scores... ", end="", flush=True)
    scores.to_csv(scores_filename, sep=',', float_format="%.2f")
    print("OK", flush=True)


if __name__ == '__main__':
    main()