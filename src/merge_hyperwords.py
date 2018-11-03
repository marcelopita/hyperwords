#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def printProgressBar (iteration, total, prefix = '', suffix = '',
                      decimals = 1, length = 50, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '░' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


def main(argv = None):
    if argv is None:
        argv = sys.argv
    H = pd.read_csv(argv[1], sep=',', header=0, index_col=0)
    orig_vocab = H.index
    vocab_map = dict(zip(orig_vocab, orig_vocab))
    beta = float(argv[2])
    min_sim = 1.0 - beta
    iteration = 0
    while True:
        iteration += 1
        progress_prefix = "Iteration " + str(iteration) + ":\t"
        vocab = H.index
        vocab_size = len(vocab)
        sims = pd.DataFrame(cosine_similarity(H), columns=vocab, index=vocab)
        drop_set = set()
        add_list = list()
        changed = False
        printProgressBar(0, vocab_size, prefix = progress_prefix, suffix = '\tComplete')
        for i in range(vocab_size):
            if vocab[i] in drop_set:
                printProgressBar(i+1, vocab_size, prefix = progress_prefix, suffix = '\tComplete')
                continue
            merge_term = None
            merge_term_sim = 0.0
            for j in range((i+1), vocab_size):
                if vocab[j] in drop_set:
                    continue
                if sims.iat[i,j] >= min_sim and sims.iat[i,j] > merge_term_sim:
                    changed = True
                    merge_term = vocab[j]
                    merge_term_sim = sims.iat[i,j]
            if merge_term is not None:
                drop_set.add(vocab[i])
                drop_set.add(merge_term)
                add_list.append((vocab[i], merge_term))
            printProgressBar(i+1, vocab_size, prefix = progress_prefix, suffix = '\tComplete')
        for w1, w2 in add_list:
            key = " ".join(sorted(set((w1 + " " + w2).split())))
            if key in H.index:
                continue
            h12 = pd.Series((H.loc[w1] + H.loc[w2])/2.0)
            h12.name = key
            for w in key.split():
                vocab_map[w] = key
            H = H.append(h12)
        H = H.drop(labels=list(drop_set))
        if not changed:
            break
    print("Saving merged hyperwords")
    H.to_csv(argv[3])
    print("Saving vocabulary mapping")
    with open(argv[4], 'wb') as f:
        pickle.dump(vocab_map, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
