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

SEQUENCE_LEN = 8        #have to compare the output of different sequence lengths
STEP = 1                #may try and compare 2/3
BATCH_SIZE = 64

def word2idx(word):
  return vector_model.wv.vocab[word].index

def idx2word(idx):
  return vector_model.wv.index2word[idx]

def print_vocabulary(words_file_path, words_set):
    words_file = io.open(words_file_path, 'w', encoding='utf8')
    for w in words_set:
        words_file.write(w+"\n")
    words_file.close()

def get_model(dropout=0.2):     #have to study the optimal dropout ratio for Bengali NLP
    model = Sequential()
    model.add(Embedding(input_dim=vocab_sz, output_dim=embed_sz, weights=[trained_wts]))
    
    model.add(Bidirectional(LSTM(units=embed_sz), input_shape=(len(sentences), SEQUENCE_LEN)))
    model.add(Dropout(dropout))
    model.add(Dense(units=vocab_sz))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
    return model

def sample(preds, temperature = 0.1):     #high temp ~ picks any, low temp ~ picks the best
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64') #prediction as numpy float64 array
    preds = np.log(preds) / temperature     #divide by temperature and take log
    exp_preds = np.exp(preds)               #take exp: e_p = e^(ln(preds)/temp)
    preds = exp_preds / np.sum(exp_preds)   #normalize
    probas = np.random.multinomial(1, preds, 1) #returns picked value
    return np.argmax(probas)    #returns index of picked value

def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    examples_file.write('\n----- Generating text after Epoch: %d\n' % epoch)
    # Randomly pick a seed sequence
    seed_index = np.random.randint(len(sentences))
    seed = (sentences)[seed_index]

    for diversity in [0.2, 0.5, 0.8]:
        sentence = seed
        examples_file.write('----- Diversity:' + str(diversity) + '\n')
        examples_file.write('----- Generating with seed:\n"' + ' '.join(sentence) + '"\n')
        examples_file.write(' '.join(sentence))

        for i in range(20):
            x_pred = np.zeros((1, SEQUENCE_LEN), dtype=np.int32)
            for t, word in enumerate(sentence):
                x_pred[0, t] = word2idx(word)

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)   #diversity ~ temperature
            next_word = idx2word(next_index)

            sentence = sentence[1:]
            sentence.append(next_word)

            examples_file.write(" "+next_word)
        examples_file.write('\n')
    examples_file.write('='*80 + '\n')
    examples_file.flush()

def generator(sentence_list, next_word_list, batch_size):
    index = 0
    while True:
        x = np.zeros((batch_size, SEQUENCE_LEN), dtype=np.int32)
        y = np.zeros((batch_size), dtype=np.int32)
        for i in range(batch_size):
            for t, w in enumerate(sentence_list[index % len(sentence_list)]):
                x[i, t] = word2idx(w)
            y[i] = word2idx(next_word_list[index % len(next_word_list)])
            index = index + 1
        yield x, y

if __name__ == "__main__":
    directory = 'F:/Minhaz/GitHubRepo/News_Gen/Minhaz_Shahadat/Code/Bengali_Word2Vec_LSTM/'
    corpus_dir = directory + 'corpus/'
    examples = directory + 'examples.txt'
    vocabulary = directory + 'vocab.txt'
    
    w_t = RegexpTokenizer("[\u0980-\u09FF']+")
    corpus = PlaintextCorpusReader(corpus_dir, r'.*\.txt', word_tokenizer=w_t)
    
    text_in_words = []
    files = corpus.fileids()
    for f in files:    
        words_in_doc = corpus.words(f)
        text_in_words.append(words_in_doc)
    text_in_words = [[re.sub(r'\d+', '<number>', word) for word in document]for document in text_in_words]
    
    words = []
    for doc in text_in_words:
        for word in doc:
            words.append(word)
    words = sorted(set(words))
    print_vocabulary(vocabulary, words)
    
    if not os.path.isdir(directory + 'checkpoints/'):
        os.makedirs(directory + 'checkpoints/')
    
    vector_model = word2vec.Word2Vec(text_in_words, size = 500, min_count = 1, window = 5)
    trained_wts = vector_model.wv.vectors
    vocab_sz, embed_sz = trained_wts.shape
    
    sentences = []
    next_words = []
    for words_in_file in text_in_words:
        for i in range(0, len(words_in_file) - SEQUENCE_LEN, STEP):
            sentences.append(words_in_file[i: i + SEQUENCE_LEN])
            next_words.append(words_in_file[i + SEQUENCE_LEN])
    print('Using sequences:', len(sentences))
        
    # will add this once we have sufficiently large data set
    #(sentences, next_words), (sentences_test, next_words_test) = shuffle_and_split_training_set(
    #    sentences, next_words)
    model = get_model()
        
    file_path = "f:/Minhaz/GitHubRepo/News_Gen/Minhaz_Shahadat/Code/Bengali_Word2Vec_LSTM/checkpoints/LSTM_Bengali_News{epoch:03d}-words%d-sequence%d" % (len(words), SEQUENCE_LEN)

    checkpoint = ModelCheckpoint(file_path)
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    callbacks_list = [checkpoint, print_callback]

    examples_file = open(examples, "w", encoding="utf-8")
    model.fit_generator(generator(sentences, next_words, BATCH_SIZE), steps_per_epoch=int(len(sentences)/BATCH_SIZE) + 1, 
                        epochs=35, callbacks=callbacks_list)




