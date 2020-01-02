# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:34:48 2019

@author: eugene
"""
import pandas as pd

train = pd.read_csv('DATA/task1_trainset.csv')
DATA = pd.read_csv('train.csv')
IDs = DATA['Id']

date = train['Created Date']

YEAR = []
MONTH = []
DAY = []

index = 0
prev_ID = IDs[index]
for ID in IDs:
    if ID != prev_ID:
        index += 1
        
    result = date[index].split('-')
    YEAR.append(result[0])
    MONTH.append(result[1])
    DAY.append(result[2])
        
    prev_ID = ID
    
df = pd.DataFrame(list(zip(YEAR, MONTH, DAY)), columns=['YEAR', 'MONTH', 'DAY'])

df.to_csv('train_date.csv', index=False)
del train
del DATA

test = pd.read_csv('DATA/task1_public_testset.csv')
DATA = pd.read_csv('test.csv')
IDs = DATA['Id']

date = test['Created Date']

YEAR = []
MONTH = []
DAY = []

index = 0
prev_ID = IDs[index]
for ID in IDs:
    if ID != prev_ID:
        index += 1
        
    result = date[index].split('-')
    YEAR.append(result[0])
    MONTH.append(result[1])
    DAY.append(result[2])
        
    prev_ID = ID
    
df = pd.DataFrame(list(zip(YEAR, MONTH, DAY)), columns=['YEAR', 'MONTH', 'DAY'])

df.to_csv('test_date.csv', index=False)
