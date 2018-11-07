#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from multiprocessing import Pool
from gensim.models import KeyedVectors
from scipy.optimize import minimize
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import sys
from _functools import partial

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
        
def load_dataset(dataset_filename):
    return pd.read_csv(dataset_filename, sep=";",
                       header=None,
                       names=["id", "class", "text"],
                       index_col=0)

def calculate_tf(corpus, vocab, binary=False):
    vectorizer = CountVectorizer(vocabulary=sorted(vocab), binary=binary)
    tf_docs = vectorizer.fit_transform(corpus)
    return pd.DataFrame(tf_docs.todense(), columns=sorted(vocab)).sort_index()

def mutual_information(H_Y, bow_docs, vocab, cos_sims, dataset_classes, alpha):
    num_docs = bow_docs.shape[0]
    docs_bool = pd.Series(num_docs*[False])
    for i in range(len(vocab)):
        if cos_sims[i] >= alpha:
            docs_bool |= bow_docs[vocab[i]].astype(bool)
    docs_bool.name = "contains_terms"
    num_docs_xi = docs_bool.sum()
    num_docs_not_xi = num_docs - num_docs_xi
    p_xi = float(num_docs_xi) / float(num_docs)
    p_not_xi = 1.0 - p_xi
    dataset_classes = pd.Series(dataset_classes.values, index=docs_bool.index, name="class")
    dc = pd.concat([docs_bool, dataset_classes], axis=1)
    p_all_y_xi = 0.0
    p_all_y_not_xi = 0.0
    for y in set(dataset_classes):
        num_docs_xi_class_y = len(dc.loc[(dc['contains_terms'] == True) & (dc['class'] == y)])
        p_y_xi = 0.0
        if num_docs_xi != 0:
            p_y_xi = float(num_docs_xi_class_y) / float(num_docs_xi)
        if p_y_xi != 0:
            p_all_y_xi += p_y_xi * np.log(p_y_xi)
            
        num_docs_not_xi_class_y = len(dc.loc[(dc['contains_terms'] == False) & (dc['class'] == y)])
        p_y_not_xi = 0.0
        if num_docs_not_xi != 0:
            p_y_not_xi = float(num_docs_not_xi_class_y) / float(num_docs_not_xi)
        if p_y_not_xi != 0:
            p_all_y_not_xi += p_y_not_xi * np.log(p_y_not_xi)
    H_Y_Xi = -p_xi * p_all_y_xi - p_not_xi * p_all_y_not_xi
    return (H_Y - H_Y_Xi)

def negative_mutual_information(H_Y, bow_docs, vocab, cos_sims, dataset_classes, alpha):
    return -mutual_information(H_Y, bow_docs, vocab, cos_sims, dataset_classes, alpha)

def optimize_alpha_exact(H_Y, bow_docs, vocab, cos_sims, dataset_classes, num_threads):
    alphas = sorted(set([round(x,1) for x in cos_sims]))
    pool = Pool(num_threads)
    func = partial(mutual_information, H_Y, bow_docs, vocab, cos_sims, dataset_classes)
    mutual_information_values = pool.map(func, alphas)
    pool.close()
    pool.join()
    return alphas[mutual_information_values.index(max(mutual_information_values))]

def optimize_alpha(H_Y, bow_docs, vocab, cos_sims, dataset_classes):
    res = minimize(negative_mutual_information, np.array([0.9]),
                   args=(H_Y, bow_docs, vocab, cos_sims, dataset_classes),
                   method="nelder-mead", options = {'xtol': 0.01, 'disp': False})
    return res.x[0]

def main(argv = None):
    if argv is None:
        argv = sys.argv
        
    # Parameters
    wv_filename = argv[1]
    hyperwords_filename = argv[2]
    vocab_filename = argv[3]
    alpha = float(argv[4])
    ds_filename = None
    num_threads = 0
    if alpha == -1.0:
        ds_filename = argv[5]
        num_threads = int(argv[6])
    
    # Load word vectors
    print("Loading word vectors... ", end='', flush=True)
    w2v_model = KeyedVectors.load_word2vec_format(wv_filename, binary=argv[1].endswith(".bin"))
    print("OK!")

    # Load dataset vocabulary
    print("Loading vocabulary... ", end='', flush=True)
    vocab_f = open(vocab_filename, 'r')
    vocab = vocab_f.readlines() 
    vocab_f.close()
    vocab = sorted([w.strip() for w in vocab])
    vocab_size = len(vocab)
    print("OK!")
    
    # Assume dynamic alpha when -1.0
    is_alpha_dynamic = False
    dataset = None
    if alpha == -1.0:
        is_alpha_dynamic = True
        # In this case, we need the dataset with classes
        print("Loading dataset... ", end='', flush=True)
        dataset = load_dataset(ds_filename)
        print("OK!")
    
    # Hyperwords
    print("Initializing hyperwords... ", end='', flush=True)
    hyperwords = pd.DataFrame(index=vocab, columns=vocab)
    hyperwords = hyperwords.fillna(0.0)
    print("OK!")
    
    # Calculation of prior H_Y and BOW, when alpha is dynamic
    H_Y = None
    bow_docs = None
    if is_alpha_dynamic:
        print("Calculating priors and BOW... ", end='', flush=True)
        num_docs = dataset.shape[0]
        H_Y = 0.0
        dataset_classes = dataset.loc[:, "class"] 
        for y in set(dataset_classes):
            p_y = float(len([c for c in dataset_classes if c==y])) / float(num_docs)
            H_Y += p_y * np.log(p_y)
        H_Y = -H_Y
        bow_docs = calculate_tf(dataset.loc[:,"text"].values.astype('U'), vocab, True)
        print("OK!")

    # Iterate over vocabulary
    print("Generation of hyperwords:", flush=True)    
    printProgressBar(0, vocab_size, prefix = 'Progress:', suffix = 'Complete')
    for i in range(vocab_size):
        
        # Calculate similarities with other word vectors
        cos_sims = []
        for j in range(vocab_size):
            if j == i:
                cos_sims.append(1.0)
                continue
            sim = 0.0
            try:
                sim = w2v_model.similarity(vocab[i], vocab[j])
            except KeyError:
                pass
            cos_sims.append(sim)
            
        # Optimize alpha for maximize mutual information
        if is_alpha_dynamic:
            alpha = optimize_alpha_exact(H_Y, bow_docs, vocab, cos_sims, dataset["class"], num_threads)
        
        # Filter by alpha    
        for j in range(vocab_size):
            if cos_sims[j] >= alpha:
                hyperwords.at[vocab[i], vocab[j]] = cos_sims[j]
                
        # Print progress
        printProgressBar(i+1, vocab_size, prefix = 'Progress:', suffix = 'Complete')
    
    # Save hyperwords to disk
    print("Saving hyperwords to disk... ", end='', flush=True)
    hyperwords.to_csv(hyperwords_filename)
    print("OK!")
        
if __name__ == '__main__':
    main()
