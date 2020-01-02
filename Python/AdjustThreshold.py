# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 09:11:50 2019

@author: eugene
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

valid1 = pd.read_csv('VALID/val_ELMO_intoDense+prepost_position(atlast).csv')
#valid2 = pd.read_csv('VALID/val_ELMO_intoDense+prepost_entity_positionNumName(atlast).csv')
valid3 = pd.read_csv('VALID/val_ELMO_intoDense+2prepost_entity_positionNumName(atlast).csv')
valid4 = pd.read_csv('VALID/val_ELMO_intoRNN+prepost_position(atlast).csv')
valid5 = pd.read_csv('VALID/val_ELMO_intoRNN+prepost_entity_positionNumName(atlast).csv')
valid6 = pd.read_csv('VALID/val_ELMO_intoRNN+2prepost_entity_positionNumName(atlast).csv')
#valid7 = pd.read_csv('VALID/val_ELMO_intoRNN+2prepost_entity_positionNumName(atlast)_CUTTO32.csv')

result1 = valid1.loc[:,'0':'5']
#result2 = valid2.loc[:,'0':'5']
result3 = valid3.loc[:,'0':'5']
result4 = valid4.loc[:,'0':'5']
result5 = valid5.loc[:,'0':'5']
result6 = valid6.loc[:,'0':'5']
#result7 = valid7.loc[:,'0':'5']

result1 = (result1+result3+result4+result6)/4
result2 = (result1+result3+result4+result5+result6)/5
result = (result1+result2)/2
#substitute to better one
result['5'] = result1['5']
result['4'] = result1['4']

label = valid4.loc[:,'0_label':'5_label']

Labels = result.columns
Thresholds = np.arange(0.0,0.8,0.01) 

for Label in Labels:
    MAX = 0
    MAX_threshold = 0
    
    y_true = label[Label+'_label']
    score = result[Label]
    for Threshold in Thresholds:
        pred = (score>Threshold).astype(int)
        f1 = f1_score(pred, y_true, average='binary')
        if f1 >= MAX:
            MAX = f1
            MAX_threshold = Threshold
        
    print('{} with threshold = {:.2f}, F1={:.5f}'.format(Label,MAX_threshold,MAX))
                                            
result_0 = (result['0']>0.40).astype(int)    #0.3 ->0.0.82998
result_1 = (result['1']>0.34).astype(int)    #0.3 ->0.60176 higher
result_2 = (result['2']>0.41).astype(int)    #0.4 ->0.70909
result_3 = (result['3']>0.33).astype(int)    #0.3 ->0.69929
result_4 = (result['4']>0.30).astype(int)    #0.3 ->0.55556 higher
result_5 = (result['5']>0.22).astype(int)    #0.3 ->0.24080
y_pred = pd.concat([result_0,result_1,result_2,result_3,result_4,result_5], axis=1)

#y_pred = (result>0.5).astype(int)
print(f1_score(y_pred, label, average='micro'))

#%% check the empty label and its sentences
#y_pred1 = (valid1.loc[:,'0':'5']>0.5).astype(int)
#y_pred2 = (valid2.loc[:,'0':'5']>0.5).astype(int)
#valid1['sum'] = y_pred1.sum(axis=1)
#valid2['sum'] = y_pred2.sum(axis=1)
#
#empty = valid1.loc[valid1['sum']==0, '0_label':'Source']
#empty.loc[:,'0_label':'5_label'].sum()
# =============================================================================
# 0_label    188 BACKGROUND
# 1_label    238 OBJECTIVES
# 2_label    339 METHODS
# 3_label    421 RESULTS
# 4_label    190 CONCLUSIONS
# 5_label     59 OTHERS
# =============================================================================

#empty = valid2.loc[valid2['sum']==0, '0_label':'Source']
#empty.loc[:,'0_label':'5_label'].sum()
# =============================================================================
# 0_label    186
# 1_label    214
# 2_label    408 METHODS
# 3_label    349 RESULTS
# 4_label    147
# 5_label     54
# =============================================================================
