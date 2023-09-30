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
pid_passages_dict = dict(zip(candidate_passages['pid'], processed_passages))

test_queries = pd.read_csv('test-queries.tsv', sep = '\t', header = None, names = ['qid', 'query'])
processed_test_queries = preprocessing(test_queries['query'], remove = True)
qid_query_dict = dict(zip(test_queries['qid'], processed_test_queries))


def dirichlet_smoothing(qid, pid_passages_dict, inverted_index, qid_query_dict, u = 50):

    query = qid_query_dict[qid]
    query_tokens_dict = Counter(query)
    query_tokens_dict = query_tokens_dict.most_common()
    #match all the same pid with the same qid in candidate-passages
    pid_list = []

    candidate_passages_qid_list = list(candidate_passages['qid'])
    candidate_passages_pid_list = list(candidate_passages['pid'])

    pid_index = [i for i,x in enumerate(candidate_passages_qid_list) if x == qid ]

    score_dict = {}
    number_words_collection = len(inverted_index.keys())
    for i in pid_index:

        pid = candidate_passages_pid_list[i]
        passage = pid_passages_dict[pid]
        number_words_passage = len(passage)
        score = 0
        for token_set in query_tokens_dict:
            token = token_set[0]
            if token in passage:
                m = inverted_index[token][pid]
            else:
                m = 0
            
            if token in list(inverted_index.keys()):
                cqi = sum(inverted_index[token].values())

            else:
                cqi = 0

            score += np.log((number_words_passage / (number_words_passage + u)) * (m / number_words_passage) + (u / (number_words_passage + u)) * (cqi / number_words_collection))

        add_dict = {pid:score}
        score_dict.update(add_dict)

    sorted_top_100 = dict(sorted(score_dict.items(), key=itemgetter(1), reverse = True)[: 100])

    return sorted_top_100



def laplace_smoothing(qid, pid_passages_dict, inverted_index, qid_query_dict):

    query = qid_query_dict[qid]
    query_tokens_dict = Counter(query)
    query_tokens_dict = query_tokens_dict.most_common()
    #match all the same pid with the same qid in candidate-passages
    pid_list = []
    candidate_passages_qid_list = list(candidate_passages['qid'])
    candidate_passages_pid_list = list(candidate_passages['pid'])

    pid_index = [i for i,x in enumerate(candidate_passages_qid_list) if x == qid ]

    score_dict = {}
    number_words_collection = len(inverted_index.keys())
    for i in pid_index:

        pid = candidate_passages_pid_list[i]
        passage = pid_passages_dict[pid]
        number_words_passage = len(passage)
        score = 0
        for token_set in query_tokens_dict:
            token = token_set[0]
            if token in passage:
                m = inverted_index[token][pid]
            else:
                m = 0
            score += np.log((m+1) / (number_words_passage + number_words_collection))

        add_dict = {pid:score}
        score_dict.update(add_dict)

    sorted_top_100 = dict(sorted(score_dict.items(), key=itemgetter(1), reverse = True)[: 100])

    return sorted_top_100

def lidstone_smoothing(qid, pid_passages_dict, inverted_index, qid_query_dict, e = 0.1):

    query = qid_query_dict[qid]
    query_tokens_dict = Counter(query)
    query_tokens_dict = query_tokens_dict.most_common()
    #match all the same pid with the same qid in candidate-passages
    pid_list = []
    candidate_passages_qid_list = list(candidate_passages['qid'])
    candidate_passages_pid_list = list(candidate_passages['pid'])

    pid_index = [i for i,x in enumerate(candidate_passages_qid_list) if x == qid ]

    N = len(pid_passages_dict)

    score_dict = {}
    number_words_collection = len(inverted_index.keys())
    for i in pid_index:


        pid = candidate_passages_pid_list[i]
        passage = pid_passages_dict[pid]
        number_words_passage = len(passage)
        score = 0
        for token_set in query_tokens_dict:
            token = token_set[0]
            if token in passage:
                m = inverted_index[token][pid]
            else:
                m = 0
            score += np.log((m+e) / (number_words_passage + e * number_words_collection))

        add_dict = {pid:score}
        score_dict.update(add_dict)

    sorted_top_100 = dict(sorted(score_dict.items(), key=itemgetter(1), reverse = True)[: 100])

    return sorted_top_100



with open('laplace.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    qid_list = list(qid_query_dict.keys())

    for qid in qid_list:
        top_100_dic = laplace_smoothing(qid, pid_passages_dict, index_inverted, qid_query_dict)
        for pid in list(top_100_dic.keys()):
            writer.writerow([qid, pid, top_100_dic[pid]])


with open('lidstone.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    qid_list = list(qid_query_dict.keys())

    for qid in qid_list:
        top_100_dic = lidstone_smoothing(qid, pid_passages_dict, index_inverted, qid_query_dict)
        for pid in list(top_100_dic.keys()):
            writer.writerow([qid, pid, top_100_dic[pid]])





with open('dirichlet.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    qid_list = list(qid_query_dict.keys())

    for qid in qid_list:
        top_100_dic = dirichlet_smoothing(qid, pid_passages_dict, index_inverted, qid_query_dict)
        for pid in list(top_100_dic.keys()):
            writer.writerow([qid, pid, top_100_dic[pid]])

time_end = time.time()
print(time_end - time_start)
#time should be 1476.3811719417572s

