# -*- coding: utf-8 -*-
from gensim.models import word2vec
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional
from keras.layers.embeddings import Embedding
import numpy as np
import io
import os
from nltk.corpus.reader import PlaintextCorpusReader
from nltk import RegexpTokenizer
import re

def word2idx(word):
  return vector_model.wv.vocab[word].index

def idx2word(idx):
  return vector_model.wv.index2word[idx]


directory = 'F:/Minhaz/GitHubRepo/News_Gen/Minhaz/Code/Bengali_Work/'
corpus_dir = directory + 'corpus/'

w_t = RegexpTokenizer("[\u0980-\u09FF']+")
corpus = PlaintextCorpusReader(corpus_dir, r'.*\.txt', word_tokenizer=w_t)

text_in_words = []
files = corpus.fileids()

for f in files:    
    words_in_doc = corpus.words(f)
    text_in_words.append(words_in_doc)

text_in_words = [[re.sub(r'\d+', '<number>', word) for word in document]for document in text_in_words]

#print('Corpus length in words:', len(text_in_words))

vector_model = word2vec.Word2Vec(text_in_words, size = 500, min_count = 1, window = 5)
trained_wts = vector_model.wv.vectors
vocab_sz, embed_sz = trained_wts.shape
words = set(corpus.words())

model = Sequential()
model.add(Embedding(input_dim=vocab_sz, output_dim=embed_sz, weights=[trained_wts]))
model.add(LSTM(units=embed_sz))
model.add(Dense(units=vocab_sz))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

