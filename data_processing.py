import string
import string
import pandas as pd
import re
import numpy as np
import os
from sklearn import preprocessing
import random
import time
import tensorflow as tf
from keras.models import Sequential

data_path = f"C:/Users/andre/Documents/TwitterData/training.1600000.processed.noemoticon.csv"
SPECIALS = []

# author: jacob ziglav
# extract target and text data for the training data
def read():
    df = pd.read_csv(data_path, names=["target", "ids", "date", "flag", "user", "text"],
                     usecols=['target', 'text'], encoding='latin-1')

    df = df[[f"target", f"text"]]
    df_test = df[0:400]
    return df_test

# process and simplify tweets
def preprocess(df_in):
    df_in['text_new'] = df_in['text'].str.lower()
    df_in['text_new'] = df_in['text_new'].apply(lambda x: re.sub('@[^\s]+', '', x))
    df_in['text_new'] = df_in['text_new'].apply(lambda x: re.sub('http[^\s]+', '', x))
    df_in['text_new'] = df_in['text_new'].str.translate(str.maketrans('', '', string.punctuation))
    print(df_in['text_new'].head())
    # done
