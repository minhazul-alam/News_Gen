# -*- coding: utf-8 -*-
from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus('f:/Minhaz/GitHubRepo/News_Gen/Minhaz/Code/Bengali_Work/word2vec_english/EC/t8') #####################3
model = word2vec.Word2Vec(sentences, size = 200)
