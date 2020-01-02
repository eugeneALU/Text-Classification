# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 22:47:10 2019

@author: eugene
"""
import pandas as pd

#%% read in data
DATA = pd.read_csv('train_removesomepunctuation_addentity.csv') # might shift to original file?

ID = DATA['Id']
#TOKEN = DATA['TOKEN']
SEN = DATA['Sentences']
LABEL = DATA.loc[:, 'BACKGROUND':'OTHERS']
POSITION = DATA['POSITION']
TOTAL_LEN = DATA['TOTAL_LEN']
LEFT = DATA.loc[:,'haveNum':'LENGTH']
LENGTH = len(ID)
del DATA

#%% mask for first and last sentences
is_first = (POSITION==1)
is_last = (POSITION==TOTAL_LEN)

#%%
# shift one element backward and discard last one
PRE_SEN = pd.concat([pd.DataFrame(['<START>']), SEN[0:-1]], ignore_index=True) 
PRE_SEN[is_first] = '<START>'

# shift one element forward and discard first
POST_SEN = pd.concat([SEN[1:], pd.DataFrame(['<END>'])], ignore_index=True)
POST_SEN[is_last] = '<END>'

#%% save the result
RESULT = pd.concat([SEN,PRE_SEN,POST_SEN,POSITION,TOTAL_LEN,LEFT], ignore_index=True, axis=1)
RESULT.columns = ['Sentences', 'PRE_Sen', 
                  'POST_Sen', 'POSITION', 'TOTAL_LEN', 'haveNum', 'haveName', 'LENGTH']
RESULT.to_csv('train_prepost_addentity.csv', index=False)

#%%TEST
DATA = pd.read_csv('test_removesomepunctuation_addentity.csv') # might shift to original file?
ID = DATA['Id']
#TOKEN = DATA['TOKEN']
SEN = DATA['Sentences']
POSITION = DATA['POSITION']
TOTAL_LEN = DATA['TOTAL_LEN']
LEFT = DATA.loc[:,'haveNum':'LENGTH']
LENGTH = len(ID)
del DATA
#% mask for first and last sentences
is_first = (POSITION==1)
is_last = (POSITION==TOTAL_LEN)

#%
# shift one element backward and discard last one
PRE_SEN = pd.concat([pd.DataFrame(['<START>']), SEN[0:-1]], ignore_index=True) 
PRE_SEN[is_first] = '<START>'

# shift one element forward and discard first
POST_SEN = pd.concat([SEN[1:], pd.DataFrame(['<END>'])], ignore_index=True)
POST_SEN[is_last] = '<END>'

#% save the result
RESULT = pd.concat([SEN,PRE_SEN,POST_SEN,POSITION,TOTAL_LEN,LEFT], ignore_index=True, axis=1)
RESULT.columns = ['Sentences', 'PRE_Sen', 
                  'POST_Sen', 'POSITION', 'TOTAL_LEN', 'haveNum', 'haveName', 'LENGTH']
RESULT.to_csv('test_prepost_addentity.csv', index=False)


#%%PRIVATE
DATA = pd.read_csv('private_removesomepunctuation.csv') # might shift to original file?
ID = DATA['Id']
#TOKEN = DATA['TOKEN']
SEN = DATA['Sentences']
POSITION = DATA['POSITION']
TOTAL_LEN = DATA['TOTAL_LEN']   
LEFT = DATA.loc[:,'haveNum':'LENGTH']
LENGTH = len(ID)
del DATA
#%% mask for first and last sentences
is_first = (POSITION==1)
is_last = (POSITION==TOTAL_LEN)

#%%
# shift one element backward and discard last one
PRE_SEN = pd.concat([pd.DataFrame(['<START>']), SEN[0:-1]], ignore_index=True) 
PRE_SEN[is_first] = '<START>'

# shift one element forward and discard first
POST_SEN = pd.concat([SEN[1:], pd.DataFrame(['<END>'])], ignore_index=True)
POST_SEN[is_last] = '<END>'

#%% save the result
RESULT = pd.concat([SEN,PRE_SEN,POST_SEN,POSITION,TOTAL_LEN,LEFT], ignore_index=True, axis=1)
RESULT.columns = ['Sentences', 'PRE_Sen', 
                  'POST_Sen', 'POSITION', 'TOTAL_LEN', 'haveNum', 'haveName', 'LENGTH']
RESULT.to_csv('private_prepost.csv', index=False)
