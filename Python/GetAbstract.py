# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 09:43:39 2019

@author: eugene
"""

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

class Sample():
    def __init__(self, row):
        self.ID = row['Id']
        self.Title = row['Title']
        self.Abstract = row['Abstract']
        self.Authors = row['Authors']
        self.Categories = row['Categories']
        self.Date = row['Created Date']
        self.label = row['Task 1']
        self.dataframe = self.get_dataframe()
        
    def get_dataframe(self):
        # Split sentences and corresponding label
        abstract_list = self.Abstract.split('$$$')
        label_lists = self.label.split(' ')
        # Split multi label
        label_list = [s.split('/')[:] for s in label_lists]
        # Store into dataframe (easier to process later)
        DICT = pd.DataFrame(list(zip(abstract_list,label_list)), columns=['Sentences', 'Label'])
        DICT['Id'] = self.ID 
        return DICT
        
train_df = pd.read_csv('./DATA/task1_trainset.csv')

Sample_obj = []
for index, row in train_df.iterrows():
    Sample_obj.append(Sample(row))
    
#Efficient way (recommendation for website) to combine several dataframes
ABSTRACT = pd.concat([sample.dataframe for sample in Sample_obj], ignore_index=True)

#CATEGORY = [['BACKGROUND','OBJECTIVES','METHODS','RESULTS','CONCLUSIONS','OTHERS']]
MLB = MultiLabelBinarizer()
OneHotLabel = MLB.fit_transform(ABSTRACT['Label'])

# Checking
print(OneHotLabel.shape)
print(MLB.classes_)

# Add to our dataframe
ABSTRACT['BACKGROUND'] = OneHotLabel[:,np.argmax(MLB.classes_=='BACKGROUND')]
ABSTRACT['OBJECTIVES'] = OneHotLabel[:,np.argmax(MLB.classes_=='OBJECTIVES')]
ABSTRACT['METHODS'] = OneHotLabel[:,np.argmax(MLB.classes_=='METHODS')]
ABSTRACT['RESULTS'] = OneHotLabel[:,np.argmax(MLB.classes_=='RESULTS')]
ABSTRACT['CONCLUSIONS'] = OneHotLabel[:,np.argmax(MLB.classes_=='CONCLUSIONS')]
ABSTRACT['OTHERS'] = OneHotLabel[:,np.argmax(MLB.classes_=='OTHERS')]
ABSTRACT.pop('Label')


class Sample():
    def __init__(self, row):
        self.ID = row['Id']
        self.Title = row['Title']
        self.Abstract = row['Abstract']
        self.Authors = row['Authors']
        self.Categories = row['Categories']
        self.Date = row['Created Date']
        self.dataframe = self.get_dataframe()
        
    def get_dataframe(self):
        # Split sentences and corresponding label
        abstract_list = self.Abstract.split('$$$')
        # Store into dataframe (easier to process later)
        DICT = pd.DataFrame(list(abstract_list), columns=['Sentences'])
        DICT['Id'] = self.ID 
        return DICT
#%%
private_df = pd.read_csv('./DATA/task1_private_testset.csv')

Sample_obj = []
for index, row in private_df.iterrows():
    Sample_obj.append(Sample(row))
    
#Efficient way (recommendation for website) to combine several dataframes
ABSTRACT = pd.concat([sample.dataframe for sample in Sample_obj], ignore_index=True)
ABSTRACT = ABSTRACT[['Id', 'Sentences']]

# Checking
print(ABSTRACT.shape)
# save
ABSTRACT.to_csv('private.csv', index=False)
