import pandas as pd
from operator import itemgetter
import numpy as np

# 특정 문서가 주어졌을 때 문서 내의 단어들의 출현 빈도를 세는 함수
def get_term_frequency(document, word_dict=None):
    if word_dict is None:
        word_dict = {}
    words = document.split()

    for w in words:
        word_dict[w] = 1+(0 if word_dict.get(w) is None else word_dict[w])
    return pd.Series(word_dict).sort_values(ascending=False)

# 문서들이 주어졌을 때 각 단어가 몇 개의 문서에서 나타났는지 세는 함수
def get_document_frequency(documents):
    dicts=[]
    vocab = set([])
    df = {}

    for d in documents:
        tf = get_term_frequency(d)
        dicts += [tf]
        vocab = vocab | set(tf.keys())
    
    for v in list(vocab):
        df[v] = 0
        for dict_d in dicts:
            if dict_d.get(v) is not None:
                df[v] += 1
    return pd.Series(df).sort_values(ascending=False)

# tf-idf 계산
def get_tfidf(docs):
    vocab = {}
    tfs = []
    for d in docs:
        vocab = get_term_frequency(d, vocab)
        tfs += [get_term_frequency(d)]
    df = get_document_frequency(docs)

    stats = []
    for word, freq in vocab.items():
        tfidfs = []
        for idx in range(len(docs)):
            if tfs[idx].get(word) is not None:
                tfidfs += [tfs[idx][word] * np.log(len(docs) / df[word])]
            else:
                tfidfs += [0]
    stats.append((word, freq, *tfidfs, max(tfidfs)))

    return pd.DataFrame(stats, columns=('word', 'frequency', 'doc1', 'doc2', 'doc3', 'max')).sort_values('max', ascending=False)

# tf 계산
def get_tf(docs):
    vocab = {}
    tfs = []
    for d in docs:
        vocab = get_term_frequency(d, vocab)
        tfs += [get_term_frequency(d)]
    
    stats = []
    for word, freq in vocab.items():
        tf_v = []
        for idx in range(len(docs)):
            if tfs[idx].get(word) is not None:
                tf_v += [tfs[idx][word]]
            else:
                tf_v += [0]
        stats.append((word, freq, *tf_v))
    return pd.DataFrame(stats, columns=('word', 'frequency', 'doc1', 'doc2', 'doc3')).sort_values('frequency', ascending=False)

