# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:16:34 2019

@author: eugene
"""
import pandas as pd 
import numpy as np

TRAIN = pd.read_csv('train.csv')

ID = TRAIN['Id'].tolist()
LEN = len(ID)
POSITION = np.zeros(LEN, dtype=int)
TOTAL_LEN = np.zeros(LEN, dtype=int)

# first entry
POSITION[0] = 1
START = 0 #store the start index, for assign TOTAL_LEN
for i in range(1,LEN):  
    # same ID
    if(ID[i]==ID[i-1]):
        POSITION[i] = POSITION[i-1] + 1  #same articale ==> POSITION+1
    else:
        TOTAL_LEN[START:i] = POSITION[i-1] #assign the TOTAL_LEN ([START:i] doesn't include index [i])
        POSITION[i] = 1 #different articale ==> reset POSITION
        START = i #start index changed, enter new article
        
TOTAL_LEN[START:i+1] = POSITION[i] #for last article
        
TRAIN['POSITION'] = POSITION
TRAIN['TOTAL_LEN'] = TOTAL_LEN
TRAIN.to_csv('train.csv', index=False)

#%% TEST
TEST = pd.read_csv('test.csv')

ID = TEST['Id'].tolist()
LEN = len(ID)
POSITION = np.zeros(LEN, dtype=int)
TOTAL_LEN = np.zeros(LEN, dtype=int)

# first entry
POSITION[0] = 1
START = 0 
for i in range(1,LEN):  
    # same ID
    if(ID[i]==ID[i-1]):
        POSITION[i] = POSITION[i-1] + 1  #same articale ==> POSITION+1
    else:
        TOTAL_LEN[START:i] = POSITION[i-1]
        POSITION[i] = 1 #different articale ==> reset POSITION
        START = i
        
TOTAL_LEN[START:i+1] = POSITION[i] 
        
TEST['POSITION'] = POSITION
TEST['TOTAL_LEN'] = TOTAL_LEN
TEST.to_csv('test.csv', index=False)

#%% plot  each label
#['BACKGROUND', 'OBJECTIVES', 'METHODS', 'RESULTS', 'CONCLUSIONS']
# =============================================================================
# import matplotlib.pyplot as plt
# LABEL = TRAIN.columns[2:8]
# 
# bins = np.arange(15) - 0.5
# for index in range(6):
#     plt.figure(index)
#     plt.hist(TRAIN.loc[TRAIN[LABEL[index]]==1,'POSITION'], bins, ec='black')
#     plt.xticks(range(15))
#     plt.xlim([0, 15])
#     plt.title(LABEL[index])
#     plt.xlabel('Position')
#     plt.show()
# =============================================================================
#%% PRIVATE
TEST = pd.read_csv('private.csv')

ID = TEST['Id'].tolist()
LEN = len(ID)
POSITION = np.zeros(LEN, dtype=int)
TOTAL_LEN = np.zeros(LEN, dtype=int)

# first entry
POSITION[0] = 1
START = 0 
for i in range(1,LEN):  
    # same ID
    if(ID[i]==ID[i-1]):
        POSITION[i] = POSITION[i-1] + 1  #same articale ==> POSITION+1
    else:
        TOTAL_LEN[START:i] = POSITION[i-1]
        POSITION[i] = 1 #different articale ==> reset POSITION
        START = i
        
TOTAL_LEN[START:i+1] = POSITION[i] 
        
TEST['POSITION'] = POSITION
TEST['TOTAL_LEN'] = TOTAL_LEN
TEST.to_csv('private.csv', index=False)