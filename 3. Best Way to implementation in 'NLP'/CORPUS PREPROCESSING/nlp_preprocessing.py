import spacy
import nltk
from nltk.tokenize import word_tokenize
from konlpy.tag import Mecab


nltk.download('punkt')
spacy_en = spacy.load('en_core_web_sm')

def tokenize(en_text):
    return [tok.text for tok in spacy_en.tokenizer(en_text)]

en_text = "A Dog Run back corner near spare bedrooms"
print(f"spaCy : {tokenize(en_text)}")
print(f'NLTK : {word_tokenize(en_text)}')
