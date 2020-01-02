# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:03:35 2019

@author: eugene
"""

import pandas as pd

submit = pd.read_csv('DATA/task1_submission.csv')

result1 = pd.read_csv('RESULT/ELMO_intoDense+prepost_position(atlast).csv')
#result2 = pd.read_csv('RESULT/ELMO_intoDense+prepost_entity_positionNumName(atlast).csv')
result3 = pd.read_csv('RESULT/ELMO_intoDense+2prepost_entity_positionNumName(atlast).csv')
result4 = pd.read_csv('RESULT/ELMO_intoRNN+prepost_position(atlast).csv')
result5 = pd.read_csv('RESULT/ELMO_intoRNN+prepost_entity_positionNumName(atlast).csv')
result6 = pd.read_csv('RESULT/ELMO_intoRNN+2prepost_entity_positionNumName(atlast).csv')

result1 = (result1+result3+result4+result6)/4
result2 = (result1+result3+result4+result5+result6)/5
result = (result1+result2)/2
#substitute to better one
result['CONCLUSIONS'] = result1['CONCLUSIONS']
result['OTHERS'] = result1['OTHERS']
#result['OBJECTIVES'] = result2['OBJECTIVES'] #threshold=0.3
#result['CONCLUSIONS'] = result2['CONCLUSIONS'] #threshold=0.3
#%% fix threshold to 0.5, and fix agrmax to empty row(should or should not?)
# =============================================================================
# result = (result>=0.5).astype(int)
# 
# for _,row in result.iterrows():
#   if row['BACKGROUND':'CONCLUSIONS'].sum() == 0:
#     L = row.idxmax()
#     row[L] = 1
# 
# submit.loc[0:len(result)-1, 'BACKGROUND':'OTHERS'] = result.loc[:,'BACKGROUND':'OTHERS']
# =============================================================================

#%% Using different threshold found on validation set
result_0 = (result['BACKGROUND']>0.40).astype(int)
result_1 = (result['OBJECTIVES']>0.34).astype(int) 
result_2 = (result['METHODS']>0.41).astype(int)
result_3 = (result['RESULTS']>0.33).astype(int)
result_4 = (result['CONCLUSIONS']>0.30).astype(int)
result_5 = (result['OTHERS']>0.22).astype(int)
y_pred = pd.concat([result_0,result_1,result_2,result_3,result_4,result_5], axis=1)

#for _,row in y_pred.iterrows():
#  if row['BACKGROUND':'CONCLUSIONS'].sum() == 0:
#    L = row.idxmax()
#    row[L] = 1

#submit.loc[0:len(result)-1, 'BACKGROUND':'OTHERS'] = y_pred.loc[:,'BACKGROUND':'OTHERS']

#%%
private1 = pd.read_csv('PRIVATE/P_ELMO_intoDense+prepost_position(atlast).csv')
#private2 = pd.read_csv('PRIVATE/P_ELMO_intoDense+prepost_entity_positionNumName(atlast).csv')
private3 = pd.read_csv('PRIVATE/P_ELMO_intoDense+2prepost_entity_positionNumName(atlast).csv')
private4 = pd.read_csv('PRIVATE/P_ELMO_intoRNN+prepost_position(atlast).csv')
private5 = pd.read_csv('PRIVATE/P_ELMO_intoRNN+prepost_entity_positionNumName(atlast).csv')
private6 = pd.read_csv('PRIVATE/P_ELMO_intoRNN+2prepost_entity_positionNumName(atlast).csv')

private1 = (private1+private3+private4+private6)/4
private2 = (private1+private3+private4+private5+private6)/5
private = (private1+private2)/2
#substitute to better one
private['CONCLUSIONS'] = private1['CONCLUSIONS']
private['OTHERS'] = private1['OTHERS']
#%% Using different threshold found on validation set
private_0 = (private['BACKGROUND']>0.40).astype(int)
private_1 = (private['OBJECTIVES']>0.34).astype(int) 
private_2 = (private['METHODS']>0.41).astype(int)
private_3 = (private['RESULTS']>0.33).astype(int)
private_4 = (private['CONCLUSIONS']>0.30).astype(int)
private_5 = (private['OTHERS']>0.22).astype(int)
y_pred_private = pd.concat([private_0,private_1,private_2,private_3,private_4,private_5], axis=1)

y = pd.concat([y_pred,y_pred_private], axis=0)
y.reset_index(drop=True,inplace=True)

#y = pd.concat([result,private], axis=0)
#y.reset_index(drop=True,inplace=True)
#GLOVE = pd.read_csv('PRIVATE/GLOVE.csv',header=None)
#GLOVE.columns = y.columns
#y = (y+GLOVE)/2
#y.iloc[:,0] = (y.iloc[:,0]>=0.44).astype(int)
#y.iloc[:,1] = (y.iloc[:,1]>=0.32).astype(int)
#y.iloc[:,2] = (y.iloc[:,2]>=0.43).astype(int)
#y.iloc[:,3] = (y.iloc[:,3]>=0.32).astype(int)
#y.iloc[:,4] = (y.iloc[:,4]>=0.30).astype(int)
#y.iloc[:,5] = (y.iloc[:,4]>=0.22).astype(int)

submit.loc[:, 'BACKGROUND':'OTHERS'] = y.loc[:,'BACKGROUND':'OTHERS']
#%%
submit.to_csv('SUBMIT/result.csv', index=False)
