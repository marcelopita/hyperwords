#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import math


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


def rebuild_all_hyperwords(H, vocab_map):
    keys = vocab_map.keys()
    H_all = pd.DataFrame(columns=sorted(keys))
    for k in keys:
        h = H.loc[vocab_map[k]]
        h.name = k
        H_all = H_all.append(h)
    return H_all.sort_index()


def calculate_tf(corpus, vocab, binary=False):
    vectorizer = CountVectorizer(vocabulary=sorted(vocab), binary=binary)
    tf_docs = vectorizer.fit_transform(corpus)
    return pd.DataFrame(tf_docs.todense(), columns=sorted(vocab)).sort_index()


def calculate_idf_hyperwords(bow, num_docs, H):
    # All |V_d_ht| => matrix of size |V|xN
    vlens = H.astype(bool).astype(int).dot(bow.T)
    print("*", end="")
    w_sums = bow.dot(H.T)
    print("*", end="")
    mus = w_sums / vlens.T
    print("*", end="")
    idfs = np.log((float(num_docs) / mus.sum(axis=0)).fillna(0.0).replace(np.inf,0))
    print("*", end="")
    return pd.DataFrame(pd.Series(idfs, index=H.index)).sort_index()
#     idf = []
#     num_docs = len(corpus)
#     vocab_size = len(vocab)
#     printProgressBar(0, vocab_size, prefix = "Generating IDF hyperwords", suffix = 'Complete')
#     i = 0
#     for idx,row in H.iterrows():
#         sum_mu = 0.0
#         for d in corpus:
#             d_words = set(d.split())
#             d_bow = []
#             for w in vocab:
#                 if w in d_words:
#                     d_bow.append(1.0)
#                 else:
#                     d_bow.append(0.0)
#             d_bow = pd.Series(d_bow, index=vocab)
#             vdh = d_bow.multiply(row, axis=0)
#             num_words = float(vdh.gt(0.0).sum(axis=0))
#             mu_hd = 0.0
#             if num_words != 0.0:
#                 mu_hd = float(vdh.sum(axis=0) / num_words)
#             sum_mu += mu_hd
#         idf.append(math.log(float(num_docs) / sum_mu))
#         i += 1
#         printProgressBar(i, vocab_size, prefix = "Generating IDF hyperwords", suffix = 'Complete')
#     return pd.DataFrame(pd.Series(idf, index=vocab)).sort_index()


def main(argv = None):
    if argv is None:
        argv = sys.argv

    dataset_filename = argv[1]
    hyperwords_filename = argv[2]
    vocab_map_filename = argv[3]
    tfidf_hyperwords_filename = argv[4]

#     dataset_filename = 'datasets/20nshort.txt'
#     hyperwords_filename = 'merged_hyperwords/hyperwords_20nshort_sg-1000_dynamic-alpha_beta-0.3.csv'
#     vocab_map_filename = 'merged_hyperwords/vocab-map_20nshort_sg-1000_dynamic-alpha_beta-0.3.pkl'
#     tfidf_hyperwords_filename = 'bag_of_hyperwords/tfidf-hyperwords_20nshort_sg-1000_dynamic-alpha_beta-0.3.csv'

    print("Loading hyperwords... ", end="", flush=True)
    H = pd.read_csv(hyperwords_filename, sep=',', header=0, index_col=0)
    print("OK!")
    print("Loading vocabulary... ", end="", flush=True)
    vocab_map = pickle.load(open(vocab_map_filename, "rb"))
    print("OK!")
    print("Rebuilding all hyperwords... ", end="", flush=True)
    H = rebuild_all_hyperwords(H, vocab_map)
    print("OK!")
    print("Loading corpus... ", end="", flush=True)
    corpus = open(dataset_filename, "r").read().splitlines()
    print("OK!")
    print("Calculating TF... ", end="", flush=True)
    vocab = sorted(vocab_map.keys())
    tf_docs = calculate_tf(corpus, vocab)
    print("OK!")
    print("Generating TF hyperwords... ", end="", flush=True)
    tf_hw = tf_docs.dot(H.transpose())
    print("OK!")
    print("Generating IDF hyperwords... ", end="", flush=True)
    idf_hw = calculate_idf_hyperwords(tf_docs.astype(bool).astype(int), len(corpus), H)
    print("OK!")
    print("Generating TF-IDF hyperwords... ", end="", flush=True)
    tfidf_hw = tf_hw.multiply(idf_hw.iloc[:,0], axis=1)
    print("OK!")
    print("Saving TF-IDF hyperwords... ", end="", flush=True)
    tfidf_hw.to_csv(tfidf_hyperwords_filename)
    print("OK!")    


if __name__ == '__main__':
    main()
