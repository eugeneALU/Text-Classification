# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:57:23 2019

@author: eugene
"""
import nltk
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.model_selection import KFold

#%% parameters
NUM_CLASS = 6

#%% load data
DATA = pd.read_csv('train_tokenize_nostem.csv')
LABEL = DATA.loc[:,'BACKGROUND':'OTHERS']
ID = DATA.loc[:, 'Id']
LENGTH = DATA.loc[:,'LENGTH']
TOKEN = DATA.loc[:,'TOKEN']
del DATA

#%% prepare TFIDF and LABEL (ignore examples with LENGTH==1)
TFIDF = TfidfVectorizer(max_features=1000)  # tokenizer=nltk.word_tokenize

train_TFIDF = TFIDF.fit_transform(TOKEN[LENGTH>1])
train_feature_name = TFIDF.get_feature_names()

train_TFIDF = train_TFIDF.todense()
train_LABEL = LABEL[LENGTH>1]

# reset index since we drop some rows
train_LABEL.reset_index(drop=True, inplace=True)

#%% Training stuff
clf = XGBClassifier(max_depth=5, n_estimators=500,learning_rate=0.1, colsample_bytree=1)
CV = KFold(n_splits=5,shuffle=True)
RESULT = {}

#%% Start training
F1 = []
for label in LABEL.columns:
    train_y = train_LABEL[label]
    
    for train, val in CV.split(train_TFIDF,train_y):
        
        clf.fit(train_TFIDF[train], train_y[train])
        
        pred = clf.predict(train_TFIDF[val])
        
        f1 = f1_score(pred, train_y[val], average='binary')
        
        F1.append(f1)
        
        
    print('{} result f1_score = {}'.format(label, np.mean(F1)))
    RESULT[label] = np.mean(F1)

#%% load test data and predict
TESTDATA = pd.read_csv('test_tokenize_nostem.csv')

test_TFIDF = TFIDF.transform(TESTDATA['TOKEN'])
test_TFIDF = test_TFIDF.todense()

print(test_TFIDF.shape)

# Retrain on whole training data
for label in LABEL.columns:
    if label != 'OTHERS':
      # init a new classifier
      clf = XGBClassifier(max_depth=5, n_estimators=500,learning_rate=0.1, colsample_bytree=1)
      # training
      train_y = train_LABEL[label]
      clf.fit(train_TFIDF, train_y)
      # Predict
      y_pred = clf.predict_proba(test_TFIDF)
      y_pred[TESTDATA['LENGTH']==1] = 0 # force the length==1 sentences to OTHERS
      # store the result 
      TESTDATA[label] = y_pred

# Assign to OTHERS
TESTDATA['OTHERS'] = 0  # init as 0
TESTDATA.loc[TESTDATA['LENGTH']==1, 'OTHERS'] = 1

for _,row in TESTDATA.iterrows():
  if row['BACKGROUND':'CONCLUSIONS'].sum() == 0:
    row['OTHERS'] = 1
    
# save to csv file
TESTDATA.to_csv('RESULT/TFIDF+Xgb.csv', index=False)
