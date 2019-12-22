# -*- coding: utf-8 -*-
import io
from nltk.corpus.reader import PlaintextCorpusReader
from nltk import RegexpTokenizer

output_file = io.open('f:/output.txt', 'w', encoding='utf-8')

w_t = RegexpTokenizer("[\u0980-\u09FF']+")
corpus = PlaintextCorpusReader('F:/Minhaz/GitHubRepo/News_Gen/Minhaz/Code/Bengali_Work/corpus/', r'.*\.txt', word_tokenizer=w_t)
files = corpus.fileids()

for f in files:
    words = corpus.words(f)
    for w in words:
        output_file.write(w+"\t")
    output_file.write('\n'+'='*30+'\n')
output_file.close()