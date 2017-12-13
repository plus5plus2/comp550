# -*- coding: utf-8 -*-
"""
Created on Thu Dec 07 15:44:19 2017

@author: Plus7
"""
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
import csv
from random import randint
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import collections  
import nltk  
import numpy as np
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.datasets import imdb

#dataset
max_features=100
maxlen=80

n_samples=0 #sample number (don't assign it manually)


#model
EMBEDDING_SIZE=80
N_NEURONS=128
OUTPUT_DIM=1
#Training
BATCH_SIZE = 1#32
N_EPOCH=1


def tokenize(string):
    #removing characters except english letter and - 
    tokenizer = RegexpTokenizer(r'[0-9a-zA-Z-]+')
    #tokenize,lemmatize and remove stopwords 
    tokens=tokenizer.tokenize(string)
    lemmatizer = WordNetLemmatizer()
    stopword = stopwords.words('english') 
    filtered=[]
    for token in tokens:
        token=lemmatizer.lemmatize(token)
        if token not in stopword:
            filtered.append(token.encode("ascii"))
    return filtered

def load_in_imdb():
    
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_train=x_train
    x_test=x_test
    y_train=y_train
    y_test=y_test
    global n_samples
    n_samples=len(x_train)+len(x_test)
    
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    return x_train, x_test, y_train, y_test
    
    



def load_in_afd():
    global n_samples
    
    n_samples=0
    word_freqs = collections.Counter()  #word frequency
     
    rawX=[]
    #dataS=[] #summary row[8]
    rawY=[]
    with open('Reviews/mini.csv') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            #dataS.append(tokenize(row[8]))
            n_samples += 1
            rawX.append(row[9])
            if int(row[6])>3:
                rawY.append(1)
            elif int(row[6])<2:
                rawY.append(0)
            else:
                if randint(0,10)>4:
                    rawY.append(1)
                else:
                    rawY.append(0)
            words = tokenize(row[9].lower())
            if len(words) > maxlen:
                words=words[:maxlen]
            for word in words:
                if len(word_freqs)<maxlen:
                    word_freqs[word] += 1
            
    
    global max_features
    #only consider the most freqent max_features words 
    max_features = min(len(word_freqs),max_features)+2  # UNK and 0  #20000
    #consider all words
    #max_features = len(word_freqs)+2  # UNK and 0  #20000
    word2index = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(max_features))}
    word2index["PAD"] = 0
    word2index["UNK"] = 1
    index2word = {v:k for k, v in word2index.items()}

    
    dataX = np.empty(n_samples,dtype=list)
    dataY = np.zeros(n_samples)
    #encode the samples
    i=0
    for raw_x,raw_y in zip(rawX,rawY):
        words = tokenize(raw_x.lower())
        if len(words) > maxlen:
            words=words[:maxlen]
        seqs = []
        for word in words:
            if word in word2index:
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["UNK"])
        dataX[i] = seqs
        dataY[i]=raw_y
        
        i += 1
    #pad sequences
    dataX = sequence.pad_sequences(dataX, maxlen=maxlen)
    
    return train_test_split(dataX, dataY, test_size=0.2,random_state=42)
    

    
x_train, x_test, y_train, y_test = load_in_imdb()

print('max_len ',maxlen)
print('nb_words ', max_features)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('Build model...')
model = Sequential()
model.add(Embedding(max_features, output_dim=EMBEDDING_SIZE,input_length=maxlen))
model.add(LSTM(N_NEURONS, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(OUTPUT_DIM, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print('Train...')
model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=N_EPOCH,
          validation_data=(x_test, y_test)
          )
score, acc = model.evaluate(x_test, y_test,
                            batch_size=BATCH_SIZE)
print('Test score:', score)
print('Test accuracy:', acc)