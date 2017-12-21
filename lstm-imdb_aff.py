# -*- coding: utf-8 -*-
"""
Created on Thu Dec 07 15:44:19 2017

@author: Plus7
"""


import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="either sl or cl",action="store",dest="model",type=str.lower)
parser.add_argument("--source", help="either imdb or afd",action="store",dest="source",type=str.lower)
parser.add_argument("--train_size", action="store", dest="train_size", type=int, default=10000)
parser.add_argument("--batch", action="store", dest="batch", type=int, default=100)
parser.add_argument("--epoch", action="store", dest="epoch", type=int, default=2)
parser.add_argument("--layers", action="store", dest="layers", type=int, default=1)
parser.add_argument("--neurals", action="store", dest="neurals", type=int, default=128)
args = parser.parse_args()
args_str = "Parsed arguments: " + str(args)
print args_str
os.system('echo "' + args_str + '" >> test.log')


from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import LSTM
import csv
from random import randint
import nltk
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
TEST_SIZE = 10000

n_samples=0 #sample number (don't assign it manually)

# Convolution
kernel_size = 5
filters = 64
pool_size = 4
#model
EMBEDDING_SIZE=80
N_NEURONS=args.neurals
OUTPUT_DIM=1
#Training
BATCH_SIZE = args.batch
N_EPOCH=args.epoch

n=20000#controlling the aff dataset size

def tokenize(string):
    #removing characters except english letter and - 
    tokenizer = RegexpTokenizer(r'[0-9a-zA-Z-]+')
    #tokenize,lemmatize and remove stopwords 
    tokens=tokenizer.tokenize(string)
    lemmatizer = WordNetLemmatizer()
    #stopword = stopwords.words('english') 
    #filtered=[]
    for token in tokens:
        token=lemmatizer.lemmatize(token)
        #if token not in stopword:
        filtered.append(token.encode("ascii"))
    return filtered

def load_in_imdb():
    
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print ("original x_train: ", x_train.shape)
    print ("original x_test: ", x_test.shape)
    x_train=x_train[:args.train_size]
    x_test=x_test[:TEST_SIZE]
    y_train=y_train[:args.train_size]
    y_test=y_test[:TEST_SIZE]
    global n_samples
    n_samples=len(x_train)+len(x_test)

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print "loaded samples from imdb, %d for training and %d for testing" % (len(x_train), len(x_test))
    return x_train, x_test, y_train, y_test
    
    



def load_in_afd():
    global max_features
    global n_samples
    
    global n
    
    n_samples=0
    word_freqs = collections.Counter()  #word frequency
     
    rawX=[]
    #dataS=[] #summary row[8]
    rawY=[]
    with open('../Reviews.csv') as f:
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
                if(len(word_freqs)<max_features):
                    word_freqs[word] += 1
            n=n-1
            if n==0:
                break   
            
    
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
        if len(words) >maxlen:
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
    x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2,random_state=42)
    print ("original x_train: ", x_train.shape)
    print ("original x_test: ", x_test.shape)
    x_train=x_train[:args.train_size]
    x_test=x_test[:TEST_SIZE]
    y_train=y_train[:args.train_size]
    y_test=y_test[:TEST_SIZE]
    return x_train, x_test, y_train, y_test
    
if args.source == "imdb":
    x_train, x_test, y_train, y_test = load_in_imdb()
elif args.source == "afd":
    x_train, x_test, y_train, y_test = load_in_afd()
else:
    print "unknown option for --source"
    import sys
    sys.exit()
    

def create_standard_model():
    global max_features
    model = Sequential()
    model.add(Embedding(max_features, output_dim=EMBEDDING_SIZE,input_length=maxlen))
    for i in range(args.layers-1):
        model.add(LSTM(N_NEURONS, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    model.add(LSTM(N_NEURONS, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(OUTPUT_DIM, activation='sigmoid'))
    
    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def create_convo_model():
    global max_features
    model = Sequential()
    model.add(Embedding(max_features, output_dim=EMBEDDING_SIZE,input_length=maxlen))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    for i in range(args.layers-1):
        model.add(LSTM(N_NEURONS, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    model.add(LSTM(N_NEURONS, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(OUTPUT_DIM, activation='sigmoid'))
    
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

print('max_len ',maxlen)
print('nb_words ', max_features)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('Build model...')

if args.model == "cl":
    model = create_convo_model()
elif args.model == "sl":
    model = create_standard_model()
else:
    print "unknown option for --model"
    import sys
    sys.exit()

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
import os
os.system('echo "%f " >> test.log' % acc)
