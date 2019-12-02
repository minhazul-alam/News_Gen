# -*- coding: utf-8 -*-
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
warnings.filterwarnings(action = 'ignore')
import gensim
from gensim.models import Word2Vec

sample = open("f:/Minhaz/GitHubRepo/News_Gen/Minhaz/Code/wl.txt")
s = sample.read()
f = s.replace("\n", " ")

data = []
for i in sent_tokenize(f):
    temp = []
    for j in word_tokenize(i):
        temp.append(j.lower())
        
    data.append(temp)

model1 = gensim.models.Word2Vec(data, min_count = 1, size = 100, window = 5)

model1.similarity('alice', 'wonderland')
model1.similarity('alice', 'machines')

model2 = gensim.models.Word2Vec(data, min_count = 1, size = 100, window = 5, sg = 1)
model2.similarity('alice', 'wonderland')
model2.similarity('alice', 'machines')
