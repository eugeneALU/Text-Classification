# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 15:35:29 2019

@author: eugene
"""
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')

sentences = train['Sentences']

train['haveNum'] = 0
train['haveName'] = 0
#train['haveHttp'] = 0
for i, s in enumerate(sentences):
    #%% number
    #  match {d}.,-{d}  ;match start of the line; match end of the line
    if re.search(r'(?:(?<= )\d+[\.\,\-]*\d*(?= )|^\d+(?= )|\d+(?=$))', s):
        train.loc[i,'haveNum'] = 1
    #%% name       
    #   do not match the first word of a line but match any uppsercase in the middle of the line
    if re.search(r'(?<= )(?:[A-Z]+[\w-]*|[\w\-]*[A-Z]+[\w\-]*)', s):
        train.loc[i,'haveName'] = 1
    #%% URL
#    if re.search(r'http.+', s):
#        train.loc[i,'haveName'] = 1    
    
#%% check result
haveNum = train.loc[train['haveNum']==1,:]
NumSum = np.sum(haveNum.loc[:,'BACKGROUND':'OTHERS'].values, axis=0)

plt.figure(0)
plt.bar(np.arange(0,6), NumSum, tick_label=haveNum.columns[2:8])
plt.title('have Number (ex:2.00;2,0;.0)')
plt.xticks(rotation=30, size=6)
plt.show()
#%% check result
haveName = train.loc[train['haveName']==1,:]
NameSum = np.sum(haveName.loc[:,'BACKGROUND':'OTHERS'].values, axis=0)

plt.figure(0)
plt.bar(np.arange(0,6), NameSum, tick_label=haveName.columns[2:8])
plt.title('have NameEntity (ex:Cifar-10;E2E)')
plt.xticks(rotation=30, size=6)
plt.show()
#%% check result
# =============================================================================
# haveHttp = train.loc[train['haveHttp']==1,:]
# HttpSum = np.sum(haveHttp.loc[:,'BACKGROUND':'OTHERS'].values, axis=0)
# 
# plt.figure(0)
# plt.bar(np.arange(0,6), HttpSum, tick_label=haveHttp.columns[2:8])
# plt.title('have URL')
# plt.xticks(rotation=30, size=6)
# plt.show()
# =============================================================================
#train.to_csv('train.csv', index=False)

#%%TEST
test = pd.read_csv('test.csv')
sentences = test['Sentences']

test['haveNum'] = 0
test['haveName'] = 0
for i, s in enumerate(sentences):
    #%% number
    # match {d}.,-{d}  ;match start of the line; match end of the line
    if re.search(r'(?:(?<= )\d+[\.\,\-]*\d*(?= )|^\d+(?= )|\d+(?=$))', s):
        test.loc[i,'haveNum'] = 1
    #%% name
    #   do not match the first word of a line but match any uppsercase in the middle of the line
    if re.search(r'(?<= )(?:[A-Z]+[\w-]*|[\w\-]*[A-Z]+[\w\-]*)', s):
        test.loc[i,'haveName'] = 1

test.to_csv('test.csv', index=False)   

#%%PRIVATE
test = pd.read_csv('private.csv')
sentences = test['Sentences']

test['haveNum'] = 0
test['haveName'] = 0
for i, s in enumerate(sentences):
    #%% number
    # match {d}.,-{d}  ;match start of the line; match end of the line
    if re.search(r'(?:(?<= )\d+[\.\,\-]*\d*(?= )|^\d+(?= )|\d+(?=$))', s):
        test.loc[i,'haveNum'] = 1
    #%% name
    #   do not match the first word of a line but match any uppsercase in the middle of the line
    if re.search(r'(?<= )(?:[A-Z]+[\w-]*|[\w\-]*[A-Z]+[\w\-]*)', s):
        test.loc[i,'haveName'] = 1

test.to_csv('private.csv', index=False)  