#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from gensim.models import KeyedVectors
import pandas as pd

def main(argv = None):
    if argv is None:
        argv = sys.argv
    
    wv_filename = argv[1]
    vocab_filename = argv[2]
    out_wv_filename = argv[3]
    
    print("Loading word vectors... ", end='', flush=True)
    w2v_model = KeyedVectors.load_word2vec_format(wv_filename, binary=argv[1].endswith(".bin"))
    print("OK!")
    
    print("Loading vocabulary... ", end='', flush=True)
    vocab_f = open(vocab_filename, 'r')
    vocab = vocab_f.readlines() 
    vocab_f.close()
    vocab = sorted([w.strip() for w in vocab])
    print("OK!")
    
    print("Extracting word vectors... ", end='', flush=True)
    out_wv = pd.DataFrame(index=vocab, columns=range(w2v_model.wv.vector_size))
    out_wv = out_wv.fillna(0.0)
    drop_words = []
    for w in vocab:
        try:
            out_wv.loc[w] = w2v_model[w]
        except KeyError:
            drop_words.append(w)
    out_wv.drop(drop_words)
    print("OK!")
    
    print("Saving extracted vectors to disk... ", end='', flush=True)
    out_wv.to_csv(out_wv_filename, sep=' ', index=True, decimal='.', header=False, float_format='%.3f')
    print("OK!")

if __name__ == '__main__':
    main()
