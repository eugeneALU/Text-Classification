# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:06:51 2019

@author: eugene
"""
#import nltk
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics


def todict(sentences):
    length = len(sentences)
    dimension = len(sentences[0])
    result = []
    for index in range(length):
        res_dct = {'f'+str(i): sentences[index][i] for i in range(dimension)} 
        result.append(res_dct)
    return result

#%% parameters
NUM_CLASS = 6

#%% load data
DATA = pd.read_csv('train_tokenize_nostem.csv')
COLUMN = DATA.columns[2:8]
LABEL = DATA.loc[:,'BACKGROUND':'OTHERS']
ID = DATA.loc[:, 'Id']
ABSTRACT_LEN = len(set(ID))  #7000
LENGTH = DATA.loc[:,'LENGTH']
TOKEN = DATA.loc[:,'TOKEN']
SENTENCE_LENGTH = DATA.loc[:,'TOTAL_LEN']
del DATA

#DATA = pd.read_csv('DATA/task1_trainset.csv')
#train_LABEL = DATA.loc[:,'Task 1']
#del DATA

TEST = pd.read_csv('test_tokenize_nostem.csv')
TEST_ID = TEST.loc[:, 'Id']
TEST_LEN = len(set(TEST_ID)) 
TEST_SENTENCE_LENGTH = TEST.loc[:,'TOTAL_LEN']
TEST_TOKEN = TEST.loc[:,'TOKEN']
del TEST

RESULT = {}
#%% prepare TFIDF and LABEL (ignore examples with LENGTH==1)
TFIDF = TfidfVectorizer(max_features=500)  # tokenizer=nltk.word_tokenize

train_TFIDF = TFIDF.fit_transform(TOKEN)
train_feature_name = TFIDF.get_feature_names()

train_TFIDF = train_TFIDF.todense().tolist()
LABEL = LABEL.astype('str')  # CRF package only receive str input
#train_LABEL = LABEL.values.tolist()

# reset index since we drop some rows
#train_LABEL.reset_index(drop=True, inplace=True)

test_TFIDF = TFIDF.transform(TEST_TOKEN)
test_TFIDF = test_TFIDF.todense().tolist()

#%% concat the feature from same abstract
X = []
start = 0
for i in range(ABSTRACT_LEN):
    end = start + SENTENCE_LENGTH[start]
    X.append(todict(train_TFIDF[start:end]))
    start = end 
    
test_X = []
start = 0
for i in range(TEST_LEN):
    end = start + TEST_SENTENCE_LENGTH[start]
    test_X.append(todict(test_TFIDF[start:end]))
    start = end 

#%% predict each label separately    
# =============================================================================
# BACKGROUND  RESULT:  0.7810559006211178
# OBJECTIVES  RESULT:  0.35077881619937695
# METHODS  RESULT:  0.5007132667617689
# RESULTS  RESULT:  0.5509573810994441
# CONCLUSIONS  RESULT:  0.3376090302719343
# OTHERS  RESULT:  0.20817843866171
# =============================================================================
for index in range(NUM_CLASS):
    print("Now Processing: ", COLUMN[index])
    # Get right label
    L = LABEL[COLUMN[index]]
    
    # concat the label from same abstract
    Y = []
    
    start = 0
    for i in range(ABSTRACT_LEN):
        end = start + SENTENCE_LENGTH[start]
        Y.append(L[start:end].tolist())
        start = end 
        
    # split to train and val
#    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.25, shuffle=False)
    
    # model 
    crf = CRF(
        algorithm='lbfgs',
        c1=0.01,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
#    crf.fit(X_train, Y_train)
    crf.fit(X, Y)
    
    # predict
#    Y_pred=crf.predict(X_val)
    Y_pred=crf.predict(test_X)
    # result for multiple prediction
#    print(COLUMN[index], ' RESULT: ',metrics.flat_f1_score(Y_val, Y_pred, average='micro',labels=crf.classes_))
    
    # flatten the result and convert back to int
    flat_list = []
    for sublist in Y_pred:
        for item in sublist:
            flat_list.append(int(item))
            
#    flat_result = []
#    for sublist in Y_val:
#        for item in sublist:
#            flat_result.append(int(item))
#            
#    print(COLUMN[index], ' RESULT: ', f1_score(flat_result,flat_list))
            
    RESULT[COLUMN[index]] = flat_list

result = pd.DataFrame.from_dict(RESULT)
result = result[COLUMN]
result.to_csv('./RESULT/CRF.csv', index=False)

#%% Submit
submit = pd.read_csv('DATA/task1_submission.csv')
submit.loc[0:len(result)-1, 'BACKGROUND':'OTHERS'] = result.loc[:,'BACKGROUND':'OTHERS']
submit.to_csv('SUBMIT/result_CRF.csv', index=False)
