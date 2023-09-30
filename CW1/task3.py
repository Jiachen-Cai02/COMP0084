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

def idf_passage(pid_passages, inverted_idx):

    idf_dict = {}
    pid_list = list(pid_passages.keys())
    N = len(pid_list)
    
    for pid in pid_list:
        passage = pid_passages[pid]
        passage_idf = {}

        for token in passage:
            n = len(inverted_idx[token])
            idf = np.log10(N / n)
            if token not in idf_dict.keys():
                passage_idf[token] = idf
        
        add_idf = {pid:passage_idf}
        idf_dict.update(add_idf)
    
    return idf_dict

IDF_dic = idf_passage(pid_passages_dict, index_inverted)

def tf_idf_passage(pid_passages, inverted_idx, idf_passages):

    tf_idf_dict = {}
    pid_list = list(pid_passages.keys())

    for pid in pid_list:
        passage = pid_passages[pid]
        passage_tf_idf = {}

        for token in passage:
            tf = inverted_idx[token][pid] / len(passage)
            if token not in passage_tf_idf.keys():
                passage_tf_idf[token] = tf * idf_passages[pid][token]
            
        add_tf_idf = {pid:passage_tf_idf}
        tf_idf_dict.update(add_tf_idf)

    return tf_idf_dict

TF_IDf_dic = tf_idf_passage(pid_passages_dict, index_inverted, IDF_dic)

test_queries = pd.read_csv('test-queries.tsv', sep = '\t', header = None, names = ['qid', 'query'])
processed_test_queries = preprocessing(test_queries['query'], remove = True)
qid_query_dict = dict(zip(test_queries['qid'], processed_test_queries))
index_inverted_query = inverted_index(test_queries['qid'], processed_test_queries)

def tf_query(qid_query_dic, index_inverted_query):

    tf_query_dic = {}
    qid_list = list(qid_query_dic.keys())
    for qid in qid_list:
        query = qid_query_dic[qid]
        query_tf = {}

        for token in query:
            tf = index_inverted_query[token][qid] / len(query)
            if token not in query_tf.keys():
                query_tf[token] = tf
            
        add_tf_idf = {qid:query_tf}
        tf_query_dic.update(add_tf_idf)

    return tf_query_dic

TF_query_dict = tf_query(qid_query_dict, index_inverted_query)

def tf_idf_query(tf_query_dic, idf_dict, qid):

    #match all the same pid with the same qid in candidate-passages
    pid_list = []
    candidate_passages_qid_list = list(candidate_passages['qid'])
    candidate_passages_pid_list = list(candidate_passages['pid'])

    tf_dict = tf_query_dic[qid]
    tf_token_set = set(list(tf_dict.keys()))

    tf_idf_query_dic = {}


    pid_index = [i for i,x in enumerate(candidate_passages_qid_list) if x == qid ]
    
    for i in pid_index:
        
        pid = candidate_passages_pid_list[i]
        idf = idf_dict[pid]
        idf_token_set = set(list(idf.keys()))

        token_intersect = tf_token_set & idf_token_set

        pid_tf_idf = {}

        for token in token_intersect:
            
            token_tf_idf = tf_dict[token] * idf[token]

            if token not in pid_tf_idf.keys():
                pid_tf_idf[token] = token_tf_idf
        
        add_dict = {pid: pid_tf_idf}
        tf_idf_query_dic.update(add_dict)
        
    return tf_idf_query_dic

def cosine_similarity(tf_idf_passage_pid, tf_idf_query_pid):

    passage_pid_token = set(list(tf_idf_passage_pid.keys()))
    query_pid_token = set(list(tf_idf_query_pid.keys()))

    token_intersect = passage_pid_token & query_pid_token

    inner_product  = 0
    for token in token_intersect:
        inner_product += tf_idf_passage_pid[token] * tf_idf_query_pid[token]
    
    Denominator = np.linalg.norm(list(tf_idf_passage_pid.values())) * np.linalg.norm(list(tf_idf_query_pid.values()))

    if inner_product == 0:
        similarity = 0
    else:   
        similarity = inner_product / Denominator

    return similarity

def find_top100(qid):

    tf_idf_query_dict = tf_idf_query(TF_query_dict, IDF_dic, qid)
    top100 = {}

    for pid in list(tf_idf_query_dict.keys()):

        tf_idf_passage_pid = TF_IDf_dic[pid]
        tf_idf_query_pid = tf_idf_query_dict[pid]

        similarity = cosine_similarity(tf_idf_passage_pid, tf_idf_query_pid)
        add_dic = {pid:similarity}
        top100.update(add_dic)

    sorted_top_100 = dict(sorted(top100.items(), key=itemgetter(1), reverse = True)[: 100])

    return sorted_top_100

with open('tfidf.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    qid_list = list(qid_query_dict.keys())

    for qid in qid_list:
        top_100_dic = find_top100(qid)
        for pid in list(top_100_dic.keys()):
            writer.writerow([qid, pid, top_100_dic[pid]])



#conduct the BM25, first derive the avdl

avdl_sum = 0
for pid in list(pid_passages_dict.keys()):
    avdl_sum += len(pid_passages_dict[pid])

avdl = avdl_sum / len(pid_passages_dict)

def BM25 (qid, inverted_index_passage, pid_passages_dict, qid_query_dict, k1 = 1.2, k2 = 100, b = 0.75):

    query = qid_query_dict[qid]
    query_tokens_dict = Counter(query)
    query_tokens_dict = query_tokens_dict.most_common()
    #match all the same pid with the same qid in candidate-passages

    candidate_passages_qid_list = list(candidate_passages['qid'])
    candidate_passages_pid_list = list(candidate_passages['pid'])

    pid_index = [i for i,x in enumerate(candidate_passages_qid_list) if x == qid ]

    N = len(pid_passages_dict)

    score_dict = {}

    for i in pid_index:

        pid = candidate_passages_pid_list[i]
        passage = pid_passages_dict[pid]
        dl = len(passage)
        K = k1 * (0.25 + 0.75*(dl / avdl))
        score = 0

        for token_set in query_tokens_dict:
            token = token_set[0]
            qf = token_set[1]

            if token in passage:
                ni = len(inverted_index_passage[token])
                fi = inverted_index_passage[token][pid]
            else:
                ni = 0
                fi = 0

            score += np.log((1/((ni + 0.5) / (N - ni + 0.5)))) * (((k1 + 1) * fi) / (K + fi)) * (((k2 + 1) * qf) / (k2 + qf))

        add_dict = {pid:score}
        score_dict.update(add_dict)

    sorted_top_100 = dict(sorted(score_dict.items(), key=itemgetter(1), reverse = True)[: 100])

    return sorted_top_100

with open('bm25.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    qid_list = list(qid_query_dict.keys())

    for qid in qid_list:
        top_100_dic = BM25(qid, index_inverted, pid_passages_dict, qid_query_dict)
        for pid in list(top_100_dic.keys()):
            writer.writerow([qid, pid, top_100_dic[pid]])


time_end = time.time()
print(time_end - time_start)
#time should be 85.6109459400177s

    






        
    
    






    

        


        





 



