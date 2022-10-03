import pandas as pd
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
    return df
def preprocess(df_in):
    df_in['text_new'] = df_in['text'].apply(lambda post: " " .join(post.lower() for word in post.split()))
    print(df_in['text_new'].head())
