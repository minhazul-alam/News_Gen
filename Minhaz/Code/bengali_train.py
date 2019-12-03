# -*- coding: utf-8 -*-
'''
Took help from many tutorials, but mostly from -
https://medium.com/coinmonks/word-level-lstm-text-generator-creating-automatic-song-lyrics-with-neural-networks-b8a1617104fb
This is a simplified version used to test performance fast
'''
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional
import numpy as np
import io
import os

SEQUENCE_LEN = 8        #have to try different sequence length
MIN_WORD_FREQUENCY = 10 #haven't used it here
STEP = 1                #have to try 2/3
BATCH_SIZE = 32         #can take larger

def print_vocabulary(words_file_path, words_set):
    words_file = io.open(words_file_path, 'w', encoding='utf8')
    for w in words_set:
        if w != "\n":
            words_file.write(w+"\n")
        else:
            words_file.write(w)
    words_file.close()

def generator(sentence_list, next_word_list, batch_size):
    index = 0
    while True:
        x = np.zeros((batch_size, SEQUENCE_LEN, len(words)), dtype=np.bool)
        y = np.zeros((batch_size, len(words)), dtype=np.bool)
        for i in range(batch_size):
            for t, w in enumerate(sentence_list[index % len(sentence_list)]):
                x[i, t, word_indices[w]] = 1
            y[i, word_indices[next_word_list[index % len(sentence_list)]]] = 1
            index = index + 1
        yield x, y

def get_model(dropout=0.2):
    print('Build model...')
    model = Sequential()
    model.add(Bidirectional(LSTM(128), input_shape=(SEQUENCE_LEN, len(words))))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(len(words)))
    model.add(Activation('softmax'))
    return model

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    examples_file.write('\n----- Generating text after Epoch: %d\n' % epoch)

    # Randomly pick a seed sequence
    seed_index = np.random.randint(len(sentences))
    seed = (sentences)[seed_index]

    for diversity in [0.3, 0.5, 0.7]:
        sentence = seed
        examples_file.write('----- Diversity:' + str(diversity) + '\n')
        examples_file.write('----- Generating with seed:\n"' + ' '.join(sentence) + '"\n')
        examples_file.write(' '.join(sentence))

        for i in range(50):
            # 1 - hot encoding of the input sequence
            x_pred = np.zeros((1, SEQUENCE_LEN, len(words)))
            for t, word in enumerate(sentence):
                x_pred[0, t, word_indices[word]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = indices_word[next_index]

            sentence = sentence[1:]
            sentence.append(next_word)

            examples_file.write(" "+next_word)
        examples_file.write('\n')
    examples_file.write('='*80 + '\n')
    examples_file.flush()
    
if __name__ == "__main__":
    corpus = 'F:/Minhaz/GitHubRepo/News_Gen/Minhaz/Code/Bengali_Work/corpus.txt'
    examples = 'F:/Minhaz/GitHubRepo/News_Gen/Minhaz/Code/Bengali_Work/examples.txt'
    vocabulary = 'F:/Minhaz/GitHubRepo/News_Gen/Minhaz/Code/Bengali_Work/vocab.txt'
    
    with io.open(corpus, encoding='utf-8') as f:
        text = f.read()
    print('Corpus length in characters:', len(text))
    
    text_in_words = [w for w in text.split(' ') if w.strip() != '' or w != '\n']
    print('Corpus length in words:', len(text_in_words))

    if not os.path.isdir('F:/Minhaz/GitHubRepo/News_Gen/Minhaz/Code/Bengali_Work/checkpoints/'):
        os.makedirs('F:/Minhaz/GitHubRepo/News_Gen/Minhaz/Code/Bengali_Work/checkpoints/')
    
    words = sorted(set(text_in_words))
    print_vocabulary(vocabulary, words)

    word_indices = dict((c, i) for i, c in enumerate(words))
    indices_word = dict((i, c) for i, c in enumerate(words))
    
    sentences = []
    next_words = []
    for i in range(0, len(text_in_words) - SEQUENCE_LEN, STEP):
        sentences.append(text_in_words[i: i + SEQUENCE_LEN])
        next_words.append(text_in_words[i + SEQUENCE_LEN])
    print('Using sequences:', len(sentences))
    
    model = get_model()
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    
    file_path = "f:/Minhaz/GitHubRepo/News_Gen/Minhaz/Code/Bengali_Work/checkpoints/LSTM_LYRICS-epoch{epoch:03d}-words%d-sequence%d" % (len(words), SEQUENCE_LEN)

    checkpoint = ModelCheckpoint(file_path)
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    callbacks_list = [checkpoint, print_callback]

    examples_file = open(examples, "w", encoding="utf-8")
    model.fit_generator(generator(sentences, next_words, BATCH_SIZE),
                        steps_per_epoch=int(len(sentences)/BATCH_SIZE) + 1,
                        epochs=50,
                        callbacks=callbacks_list)