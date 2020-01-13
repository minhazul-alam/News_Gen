# -*- coding: utf-8 -*-
import io
import re

output_file = io.open('f:/output.txt', 'w', encoding='utf-8')
in_file = open('F:/Minhaz/GitHubRepo/News_Gen/Minhaz_Shahadat/Code/Bengali_Word2Vec_LSTM/corpus/1.txt', 'r', encoding='utf-8')
lines = in_file.read()

re.sub(r'\u09F7', ' । ', lines)

print(lines)