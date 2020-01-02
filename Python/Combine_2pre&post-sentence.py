# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 13:51:59 2019

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

# shift two element backward and discard last one
PREPRE_SEN = pd.concat([pd.DataFrame(['<START>']), PRE_SEN[0:-1]], ignore_index=True) 
PREPRE_SEN[is_first] = '<START>'

# shift one element forward and discard first
POST_SEN = pd.concat([SEN[1:], pd.DataFrame(['<END>'])], ignore_index=True)
POST_SEN[is_last] = '<END>'

# shift two element forward and discard first
POSTPOST_SEN = pd.concat([POST_SEN[1:], pd.DataFrame(['<END>'])], ignore_index=True)
POSTPOST_SEN[is_last] = '<END>'

#%% save the result
RESULT = pd.concat([SEN,PRE_SEN,PREPRE_SEN,POST_SEN,POSTPOST_SEN,POSITION,TOTAL_LEN,LEFT], ignore_index=True, axis=1)
RESULT.columns = ['Sentences', 'PRE_Sen','PREPRE_Sen', 'POST_Sen', 'POSTPOST_Sen', 
                  'POSITION', 'TOTAL_LEN', 'haveNum', 'haveName', 'LENGTH']
RESULT.to_csv('train_2prepost_addentity.csv', index=False)

#%% Same thing to test data
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

# shift two element backward and discard last one
PREPRE_SEN = pd.concat([pd.DataFrame(['<START>']), PRE_SEN[0:-1]], ignore_index=True) 
PREPRE_SEN[is_first] = '<START>'

# shift one element forward and discard first
POST_SEN = pd.concat([SEN[1:], pd.DataFrame(['<END>'])], ignore_index=True)
POST_SEN[is_last] = '<END>'

# shift two element forward and discard first
POSTPOST_SEN = pd.concat([POST_SEN[1:], pd.DataFrame(['<END>'])], ignore_index=True)
POSTPOST_SEN[is_last] = '<END>'

#% save the result
RESULT = pd.concat([SEN,PRE_SEN,PREPRE_SEN,POST_SEN,POSTPOST_SEN,POSITION,TOTAL_LEN,LEFT], ignore_index=True, axis=1)
RESULT.columns = ['Sentences', 'PRE_Sen','PREPRE_Sen', 'POST_Sen', 'POSTPOST_Sen', 
                  'POSITION', 'TOTAL_LEN', 'haveNum', 'haveName', 'LENGTH']
RESULT.to_csv('test_2prepost_addentity.csv', index=False)

#%% PRIVATE
DATA = pd.read_csv('private_removesomepunctuation_addentity.csv') # might shift to original file?
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

# shift two element backward and discard last one
PREPRE_SEN = pd.concat([pd.DataFrame(['<START>']), PRE_SEN[0:-1]], ignore_index=True) 
PREPRE_SEN[is_first] = '<START>'

# shift one element forward and discard first
POST_SEN = pd.concat([SEN[1:], pd.DataFrame(['<END>'])], ignore_index=True)
POST_SEN[is_last] = '<END>'

# shift two element forward and discard first
POSTPOST_SEN = pd.concat([POST_SEN[1:], pd.DataFrame(['<END>'])], ignore_index=True)
POSTPOST_SEN[is_last] = '<END>'

#% save the result
RESULT = pd.concat([SEN,PRE_SEN,PREPRE_SEN,POST_SEN,POSTPOST_SEN,POSITION,TOTAL_LEN,LEFT], ignore_index=True, axis=1)
RESULT.columns = ['Sentences', 'PRE_Sen','PREPRE_Sen', 'POST_Sen', 'POSTPOST_Sen', 
                  'POSITION', 'TOTAL_LEN', 'haveNum', 'haveName', 'LENGTH']
RESULT.to_csv('private_2prepost_addentity.csv', index=False)
