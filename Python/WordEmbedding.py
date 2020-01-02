# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 20:29:18 2019

@author: eugene
"""
import spacy
import pickle
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from multiprocessing import Pool
from nltk.tokenize import word_tokenize

DATA = pd.read_csv('train_tokenize.csv')
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
from ast import literal_eval
print(type(DATA.loc[0,'TOKEN']))

# convert str back to correct list type, this happens since we store the file into .csv
DATA['TOKEN'] = DATA['TOKEN'].apply(literal_eval)
print(type(DATA.loc[0,'TOKEN']))
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#%% construct word_dict with multi-process to speed up

#run before you want to excute the code in Ipython console for create multiprocess
if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

DATA = pd.read_csv('train.csv')
n_workers = 4

chunks = [
        ' '.join( DATA['Sentences'][i:i + len( DATA['Sentences']) // n_workers])
        for i in range(0, len( DATA['Sentences']), len( DATA['Sentences']) // n_workers)
        ]

pool = Pool(processes=n_workers)
result = pool.map_async(word_tokenize, chunks)  # tokenize using nltk.word_tokenize   
words = set(sum(result.get(), []))

word_dict = {'<pad>':0}
for word in words:
    word_dict[word]=len(word_dict)
#%% Train w2v
TRAIN_corpus = DATA['TOKEN'].values
# setting
vector_dim = 64
window_size = 5
min_count = 1
training_iter = 20

# model
word2vec_model = Word2Vec(sentences=TRAIN_corpus, 
                          size=vector_dim, window=window_size, 
                          min_count=min_count, iter=training_iter)

#%% Embedding (gensim)
model_path = "./GoogleNews-vectors-negative300.bin.gz"
w2v_google_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

F = open('gensim_w2v.pkl', 'ab')
for row in DATA.loc[0:2,'TOKEN']:
  # numpy to store the result
  result = np.zeros((len(row),300))
  for i, token in enumerate(row):
  # using try in case encounter unseen word
    try:
      result[i] = w2v_google_model[token]
    except: continue
        
  pickle.dump(result, F)
F.close()

# load the data
a=[]
with open('gensim_w2v.pkl', 'rb') as F:
    while True:       # keep reading till the end of the file
        try:
            a.append(pickle.load(F))
        except EOFError:
            break

#%% Embedding (spacy)
# download the file
# !python -m spacy download en_core_web_md
model = spacy.load('en_core_web_md')

F = open('spacy_w2v.pkl', 'ab')
for row in DATA.loc[:,'TOKEN']:
    result = np.zeros((len(row),300))
    print(len(row))
    for i, token in enumerate(row):
        result[i] = model(token).vector
        
    pickle.dump(result, F)
F.close()

a=[]
with open('spacy_w2v.pkl', 'rb') as F:
    while True:
        try:
            a.append(pickle.load(F))
        except EOFError:
            break