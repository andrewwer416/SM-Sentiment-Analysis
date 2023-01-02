#data handling/preprocessing packages
import string
import pandas as pd
import re
import nltk
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from nltk.corpus import wordnet, stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
from deep_translator import GoogleTranslator
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras import regularizers
import numpy as np
import os
import ast

#data_path = f"C:/Users/andre/Documents/TwitterData/training.1600000.processed.noemoticon.csv"
data_path = f"./lemmatized_tweets.csv"
SPECIALS = []
# extract target and text data for the training data
def read():
    #df = pd.read_csv(data_path, names=["target", "ids", "date", "flag", "user", "text"],
                     #usecols=['target', 'text'], encoding='latin-1')
    df = pd.read_csv(data_path, header=0, low_memory=False)
    print(df.head())
    df["text"] = df["text"].apply(lambda x: ast.literal_eval(x))
    #df = df[[f"target", f"text"]]
    return df
stop_words = stopwords.words('english')
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

def detokenize(text):
    return TreebankWordDetokenizer().detokenize(text)

# process and simplify tweets
def text_preprocess(df_in):
    df_in['target'] = (df_in['target'] / 4).astype('int')  # change to 0, 1
    df_in['text'] = df_in['text'].str.lower()  # lowercase
    df_in['text'] = df_in['text'].apply(lambda x: re.sub('@[^\s]+', '', x))  # remove mentions
    df_in['text'] = df_in['text'].apply(lambda x: re.sub('http[^\s]+', '', x))  # remove http links
    df_in['text'] = df_in['text'].apply(lambda x: re.sub('www.[^\s]+', '', x))  # remove www links
    df_in['text'] = df_in['text'].str.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    # apply language translation
    #df_in['text'] = df_in['text'].apply(lambda x: GoogleTranslator(source='auto', target='en').translate(text=x))
    df_in['text'] = df_in['text'].apply(lemmatize)
    df_in['text'] = df_in['text'].apply(lambda x: [w for w in x if w not in stop_words])
    df_in.to_csv('lemmatized_tweets.csv', index=False)
    df_in['tokens'] = df_in['text'].apply(lambda x: detokenize(x))
    data = df_in['tokens'].to_numpy()
    labels = df_in['target'].to_numpy()
    #labels = tf.keras.utils.to_categorical(labels, 2, dtype="float32")
    #print(len(labels))
    return [data, labels]

def preprocess(df_in):
    [data, labels] = preprocess_lem(df_in)

    vocab_size = 40000
    max_length = 20
    trunc_type = 'post'
    oov_tok = '<OOV>'
    padding_type = 'post'
    #split data before fitting
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.01, train_size=0.99,
                                                      random_state=0)
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(X_train)
    print(X_train[0:10])
    print(y_train[0:10])
    #word_index = tokenizer.word_index
    train_sequences = tokenizer.texts_to_sequences(X_train)
    train_padded = pad_sequences(train_sequences, maxlen=max_length, truncating=trunc_type)
    test_sequences = tokenizer.texts_to_sequences(X_val)
    test_padded = pad_sequences(test_sequences, maxlen=max_length)
    return [train_padded, test_padded, y_train, y_val]

def preprocess_lem(df_in):
    df_in['tokens'] = df_in['text'].apply(lambda x: detokenize(x))
    data = df_in['tokens'].to_numpy()
    labels = df_in['target'].to_numpy()
    return [data, labels]