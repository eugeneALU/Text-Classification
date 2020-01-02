# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:27:02 2019

@author: eugene
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

DATA = pd.read_csv('train.csv')
CONTAIN_NAN = DATA[DATA.Sentences.isna()]

CATE = DATA.loc[:, 'BACKGROUND':'OTHERS']

MULTI = CATE[CATE.sum(axis=1)>1]

print('Is there any multilabel accompany OTHERS:', (MULTI.OTHERS > 0).any())

# Drop OTHERS since no one is pair with it
ba = MULTI[MULTI.BACKGROUND>0].sum().values[:-1]
obj = MULTI[MULTI.OBJECTIVES>0].sum().values[:-1]
meth = MULTI[MULTI.METHODS>0].sum().values[:-1]
res = MULTI[MULTI.RESULTS>0].sum().values[:-1]
con = MULTI[MULTI.CONCLUSIONS>0].sum().values[:-1]

whole = [ba,obj,meth,res,con]

col = ['BACKGROUND','OBJECTIVES','METHODS','RESULTS','CONCLUSIONS']
mask = np.ones_like(whole)
mask[np.triu_indices_from(mask)] = False

fig , ax = plt.subplots()
ax = sns.heatmap(whole,annot=True, fmt="d",linewidths=.5,mask=mask,
                 cmap=sns.light_palette("navy"))
ax.set_xticklabels(col,rotation=0, size=8) 
ax.set_yticklabels(col,rotation=0, size=8) 

COUNT = np.zeros((DATA.shape[0]))
for i in range(DATA.shape[0]):
    if type(DATA.Sentences[i]) == str:
        COUNT[i] = len(DATA.Sentences[i].split(' '))  
        
fig = plt.figure()
plt.plot(COUNT, 'bo')
plt.show()

COUNT0 = DATA[COUNT==0]
count0_others = COUNT0.OTHERS.sum()
COUNT1 = DATA[COUNT==1]
count1_others = COUNT1.OTHERS.sum()
