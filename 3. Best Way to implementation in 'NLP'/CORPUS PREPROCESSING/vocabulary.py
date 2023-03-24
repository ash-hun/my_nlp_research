import urllib.request
import pandas as pd
from eunjeon import Mecab
from nltk import FreqDist
import numpy as np
import matplotlib.pyplot as plt

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
data = pd.read_table('ratings.txt')
print(data[:10])
print('전체 샘플의 수 : {}'.format(len(data)))

sample_data = data[:100]
sample_data['document'] = sample_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣]", "")

stop_words = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
tokenizer = Mecab()
tokenized = []
for sentence in sample_data['document']:
    temp = tokenizer.morphs(sentence)
    temp = [word for word in temp if not word in stop_words]
    tokenized.append(temp)
# print(tokenized[:10])
vocab = FreqDist(np.hsatck(tokenized))
# print(f'단어 집합의 크기 : {len(vocab)}')
vocab_size = 500
vocab = vocab.most_common(vocab_size) # 상위 vocab_size개의 단어만 보존

word_to_index = {word[0]:index+2 for index, word in enumerate(vocab)}
word_to_index['pad'] = 1
word_to_index['unk'] = 0

encoded = []
for line in tokenized:
    temp = []
    for w in line:
        try:
            temp.append(word_to_index[w])
        except KeyError:
            temp.append(word_to_index['unk'])
    encoded.append(temp)

max_len = max(len(l) for l in encoded)
for line in encoded:
    if len(line) < max_len:
        line += [word_to_index['pad']] * (max_len - len(line))