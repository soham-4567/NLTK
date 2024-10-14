import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import re
import nltk
import gensim
from gensim.models import word2vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
%matplotlib inline
nltk.download('stopwords')

data = pd.read_csv('r"C:\Users\Intern\Desktop\Myques.csv"',sep=',',encoding='utf-8',error_bad_lines=False)
data.head()

data.columns

STOP_WORDS = nltk.corpus.stopwords.words()

def clean_sentence(val):
"remove chars that are not letters or numbers, downcase, then remove stop words"
regex = re.compile('([^\s\w]|_)+')
sentence = regex.sub('', val).lower()
sentence = sentence.split(" ")
for word in list(sentence):
if word in STOP_WORDS:
sentence.remove(word)
sentence = " ".join(sentence)
return sentence

def clean_dataframe(data):
"drop nans, then apply 'clean_sentence' function to Description"
data = data.dropna(how="any")
for col in ['Questions']:
data[col] = data[col].apply(clean_sentence)
return data

data = clean_dataframe(data)
data.head(5)

def build_corpus(data):
"Creates a list of lists containing words from each sentence"
corpus = []
for col in ['']:
for sentence in data[col].iteritems():
word_list = sentence[1].split(" ")
corpus.append(word_list)
return corpus

corpus = build_corpus(data)
corpus[0:10]

model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=2, workers=4)
model.wv['luxurious']

