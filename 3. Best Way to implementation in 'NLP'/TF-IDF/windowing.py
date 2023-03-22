from collections import defaultdict

import pandas as pd

def get_context_counts(lines, w_size=2):
    co_dict = defaultdict()

    for line in lines:
        words = line.split()

        for i, w in enumerate(words):
            for c in words[i - w.size:i+w_size]:
                if w != c:
                    co_dict[(w,c)] += 1
    
    return pd.Series(co_dict)

def co_occurrence(co_dict, vocab):
    data = []

    for word1 in vocab:
        row = []
        for word2 in vocab:
            try:
                count = co_dict[(word1, word2)]
            except KeyError:
                count = 0
            row.append(count)
        data.append(row)

    return pd.DataFrame(data, index=vocab, columns=vocab)

