import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import re
import string
from operator import itemgetter
import csv
from collections import Counter
import time

time_start = time.time()

def loadtext(filepath):
    with open(filepath, 'r') as f:
        txt = []
        for line in f.readlines():
            txt.append(line.strip().lower())
        
    return txt

def preprocessing(text, remove):
    #initial setting
    lemmatizer = WordNetLemmatizer()
    stemmer = SnowballStemmer('english')
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\s+', gaps = True)
    processed_txt = []

    for line in text:
        sentence = []
        #remove the punctuation
        table = str.maketrans(dict.fromkeys(string.punctuation))
        line = line.translate(table)

        line = re.sub(r'[^a-zA-Z\s]', u' ', line, flags = re.UNICODE)

        line_token = tokenizer.tokenize(line)

        #remove stop words
        if (remove == True):
            line_token = [w for w in line_token if not w in stop_words]
        

        for word in line_token:
            word = lemmatizer.lemmatize(word)
            word = stemmer.stem(word)
            sentence.append(word)

        processed_txt.append(sentence)
    
    return processed_txt

def inverted_index(pid_data, processed_data):

    dic = {}
    
    for i in range(len(pid_data)):
        processed_passage = processed_data[i]

        for token in processed_passage:
            token_num = processed_passage.count(token)

            if token not in dic.keys():
                dic[token] = {pid_data[i]: token_num}
            else:
                add_num = {pid_data[i]: token_num}
                dic[token].update(add_num)

    return dic

candidate_passages = pd.read_csv('candidate-passages-top1000.tsv', sep = '\t', header = None, names = ['qid', 'pid', 'query', 'passage'])
processed_passages = preprocessing(candidate_passages['passage'], remove = True)

index_inverted = inverted_index(candidate_passages['pid'], processed_passages)
print(len(index_inverted.keys()))
time_end = time.time()
print(time_end - time_start)