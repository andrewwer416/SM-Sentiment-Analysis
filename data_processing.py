#data handling/preprocessing packages
import string
import pandas as pd
import re
import nltk
from nltk.corpus import wordnet
from deep_translator import GoogleTranslator
import numpy as np
import os

from sklearn import preprocessing
import random
import time
import tensorflow as tf
from keras.models import Sequential

data_path = f"C:/Users/andre/Documents/TwitterData/training.1600000.processed.noemoticon.csv"
SPECIALS = []

# extract target and text data for the training data
def read():
    df = pd.read_csv(data_path, names=["target", "ids", "date", "flag", "user", "text"],
                     usecols=['target', 'text'], encoding='latin-1')

    df = df[[f"target", f"text"]]
    df_test = df[0:400]
    return df_test

#lemmatizing objects
lemm = nltk.stem.WordNetLemmatizer()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()

#used to determine type of word to lemmatize better
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)
def lemmatize(text):
    return [lemm.lemmatize(w, get_wordnet_pos(w)) for w in w_tokenizer.tokenize(text)]
# process and simplify tweets
def preprocess(df_in):
    print(df_in[0:10])
    df_in['target'] = (df_in['target'] / 2 - 1).astype('int') # change to -1, 0, 1 scale
    df_in['text'] = df_in['text'].str.lower() #lowercase
    df_in['text'] = df_in['text'].apply(lambda x: re.sub('@[^\s]+', '', x)) #remove mentions
    df_in['text'] = df_in['text'].apply(lambda x: re.sub('http[^\s]+', '', x)) #remove http links
    df_in['text'] = df_in['text'].apply(lambda x: re.sub('www.[^\s]+', '', x)) #remove www links
    df_in['text'] = df_in['text'].str.translate(str.maketrans('', '', string.punctuation)) #remove punctuation
    #apply language translation
    df_in['text'] = df_in['text'].apply(lambda x: GoogleTranslator(source='auto', target='en').translate(text=x))
    df_in['text'] = df_in['text'].apply(lemmatize)
    print(df_in[0:10])

