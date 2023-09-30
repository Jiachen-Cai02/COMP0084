import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import re
import string
import matplotlib.pyplot as plt


def loadtext(filepath):
    with open(filepath, 'r') as f:
        txt = []
        for line in f.readlines():
            txt.append(line.strip().lower())
        
    return txt

data = loadtext('passage-collection.txt')


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
            #word = lemmatizer.lemmatize(word)
            #word = stemmer.stem(word)
            sentence.append(word)

        processed_txt.append(sentence)
    
    return processed_txt

processed = preprocessing(data, False)

#create the inverted index
def inverted_index(processed_data):

    dic = {}
    for line in processed_data:
        for word in line:
            if word in dic.keys():
                dic[word] += 1
            else:
                dic[word] = 1
    return dic

index = inverted_index(processed)
words = list(index.keys())
counts = list(index.values())
frequency = counts/(np.sum(counts))
frequency_list = frequency.tolist()

#display the results

df = pd.DataFrame({
    'word': words,
    'frequency': frequency_list
})
df = df.sort_values(by=['frequency'], ascending = False)
ranking = list(range(1, len(words) + 1))
frequency_list.sort(reverse = True)
ranking_frequency =  [x * y for x, y in zip(ranking, frequency_list)]
df.insert(1, 'rank', ranking)
df.insert(3, 'rank*frequency', ranking_frequency)

#implement the Zipf's law

df['Zipf frequency'] = 1 / (df['rank'] * sum([i ** (-1) for i in range(1, len(words) + 1)]))

#select and display top 1000 terms
df_1000 = df.head(1000)
df_1000.plot(x = 'rank', y = ['frequency', 'Zipf frequency'], style = ['-', '-.'], xlabel = 'Term frequency ranking', ylabel = 'Term prob of occurrence', title = 'top1000 frequent words')
plt.show()

#plot the loglog plot
df.plot(x = 'rank', y = ['frequency', 'Zipf frequency'], style = ['-', '-.'], xlabel = 'log term frequency ranking', ylabel = 'log erm prob of occurrence', title = 'All words', loglog=True)
plt.show()
        

        

        



