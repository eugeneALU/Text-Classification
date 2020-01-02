# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 09:43:39 2019

@author: eugene
"""
import numpy as np
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

#function to clean the word of any punctuation or special characters
def cleanPunc(sentence):
    #remove word in ()
    cleaned = re.sub(r'\([A-Z|a-z]+\)',r'',sentence)
    #preserve some important punctuation and add a space between them and words
    cleaned = re.sub(r"([.!?])", r" \1", cleaned)
    #remove other punctuation
    cleaned = re.sub(r'[\'|"|#|:|;|,|%|<|>|\(|\)|\^|\||/|\[|\]|&|*|+|=|@|`|~]',r'',cleaned)
    #remove space \n in the end of the sentences
    cleaned = cleaned.strip()
    #remove numbers and words right follow numbers like XXcm(or XX-cm, XX-) but XX cm won't be remove
    cleaned = re.sub(r'\d+\-\w*|\d+\w+|\d+',r'',cleaned)
    #remove - in -XXX or XXX- or -- but no XXX-XXX
    cleaned = re.sub(r'(?<=\w)-(?!\w)|(?<!\w)-(?=\w)|--|(?<!\w)-(?!\w)',r'',cleaned)
    #remove space and - in the start or end of the sentences again
    cleaned = cleaned.strip(' -')
    #incase there are two space 
    cleaned = cleaned.replace('  ',' ')   
    return cleaned

def cleanPunc2(sentence):
    #remove word in ()
    cleaned = re.sub(r'\([A-Z|a-z|0-9]+\)',r'',sentence)
    #preserve some important punctuation and add a space between them and words
    cleaned = re.sub(r"([.!?])", r" \1 ", cleaned)
    #remove other punctuation
    cleaned = re.sub(r'[^a-zA-Z0-9?.!]+',r' ',cleaned)
    #restore special term 
    cleaned = re.sub(r'e \. g \.', "e.g.", cleaned) 
    cleaned = re.sub(r'a \. k \. a .', "a.k.a.", cleaned) 
    cleaned = re.sub(r'i \. e \.', "i.e.", cleaned)
    cleaned = re.sub(r'\. \. \.', "...", cleaned)
    cleaned = re.sub(r'(\d+) \. (\d*)', r"\1.\2", cleaned) 
    #incase there are many spaces 
    cleaned = re.sub(r'[" "]+', " ", cleaned)  
    #remove space \n in the end of the sentences
    cleaned = cleaned.strip()
    #remove space and - in the start or end of the sentences again
    cleaned = cleaned.strip(' -')
    return cleaned

def cleanPunc3(sentence):
    #remove word in ()
    cleaned = re.sub(r'\([A-Z|a-z|0-9]+\)',r'',sentence)
    #rule-based, replace the nameentity to certain entity (EX: 3D P2P CNN Tokenizer) 
    #                   -- word with uppercase but appears in the middle of the sentence
    cleaned = re.sub(r'(?<= )(?:[A-Z]+[\w-]*|[\w\-]*[A-Z]+[\w\-]*)', '<NAME>', cleaned)
    #replace the number to certain entity (EX: 2000 2.05 2,000 2-5)
    cleaned = re.sub(r'(?:(?<= )\d+[\.\,\-]*\d*(?= )|^\d+(?= )|\d+(?=$))', '<NUMBER>', cleaned)
    #preserve some important punctuation and add a space between them and words
    cleaned = re.sub(r"([.!?])", r" \1 ", cleaned)
    #remove other punctuation
    cleaned = re.sub(r'[^a-zA-Z0-9?.!]+',r' ',cleaned)
    #restore special term 
    cleaned = re.sub(r'e \. g \.', "e.g.", cleaned) 
    cleaned = re.sub(r'a \. k \. a .', "a.k.a.", cleaned) 
    cleaned = re.sub(r'i \. e \.', "i.e.", cleaned)
    cleaned = re.sub(r'\. \. \.', "...", cleaned)
    cleaned = re.sub(r'(\d+) \. (\d*)', r"\1.\2", cleaned) 
    #incase there are many spaces 
    cleaned = re.sub(r'[" "]+', " ", cleaned)  
    #remove space \n in the end of the sentences
    cleaned = cleaned.strip()
    #remove space and - in the start or end of the sentences again
    cleaned = cleaned.strip(' -')
    return cleaned

def removeStopWords(sentence):
    stop_words = set(stopwords.words('english'))
    re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
    return re_stop_words.sub("", sentence)

def stemming(sentence):
    stemmer = SnowballStemmer("english")
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

DATA = pd.read_csv('./train.csv')

''' Remove special punctuation and lower and stemming'''
#DATA['Sentences'] = DATA['Sentences'].str.lower()
DATA['Sentences'] = DATA['Sentences'].apply(cleanPunc3)
#DATA['Sentences'] = DATA['Sentences'].apply(stemming)

''' Count length'''
if DATA.Sentences.notna().all():
    COUNT = np.zeros((DATA.shape[0]))
    for i in range(DATA.shape[0]):
        COUNT[i] = len(DATA.Sentences[i].split(' '))  
DATA['LENGTH'] = COUNT
        
## replace empty sentences with <UNK> token
DATA.loc[DATA['Sentences']=='','Sentences'] = '<UNK>'
## count the sentences with only Length==1/2 -- could be one of the features
#COUNT1 = DATA[COUNT==1]
#COUNT2 = DATA[COUNT==2]

## Tokenize
#DATA['TOKEN'] = DATA.apply(lambda row: word_tokenize(row['Sentences']), axis=1)    # using tokenizer
#DATA['TOKEN'] = DATA.apply(lambda row: row['Sentences'].split(), axis=1)           # or just split with space

DATA.to_csv('train_removesomepunctuation_addentity.csv', index=False)

#%% TEST
TEST = pd.read_csv('test.csv')
''' Remove special punctuation and lower and stemming'''
#TEST['Sentences'] = TEST['Sentences'].str.lower()
TEST['Sentences'] = TEST['Sentences'].apply(cleanPunc3)
#TEST['Sentences'] = TEST['Sentences'].apply(stemming)

if TEST.Sentences.notna().all():
    COUNT = np.zeros((TEST.shape[0]))
    for i in range(TEST.shape[0]):
        COUNT[i] = len(TEST.Sentences[i].split(' '))  
TEST['LENGTH'] = COUNT

## replace empty sentences with <UNK> token
TEST.loc[TEST['Sentences']=='','Sentences'] = '<unk>'
## Tokenize
#TEST['TOKEN'] = TEST.apply(lambda row: word_tokenize(row['Sentences']), axis=1)
#TEST['TOKEN'] = TEST.apply(lambda row: row['Sentences'].split(), axis=1)

TEST.to_csv('test_removesomepunctuation_addentity.csv', index=False)

#%% PRIVATE
PRIVATE = pd.read_csv('private.csv')
''' Remove special punctuation and lower and stemming'''
PRIVATE['Sentences'] = PRIVATE['Sentences'].apply(cleanPunc2)

if PRIVATE.Sentences.notna().all():
    COUNT = np.zeros((TEST.shape[0]))
    for i in range(PRIVATE.shape[0]):
        COUNT[i] = len(PRIVATE.Sentences[i].split(' '))  
PRIVATE['LENGTH'] = COUNT

## replace empty sentences with <UNK> token
PRIVATE.loc[PRIVATE['Sentences']=='','Sentences'] = '<unk>'
# save
PRIVATE.to_csv('private_removesomepunctuation.csv', index=False)