from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import pandas as pd
import re
import numpy as np
from nltk import ngrams
import tensorflow as tf
from tensorflow.contrib import learn

x_text = pd.Series(["the dog's ran", 'the cat ran', 'the bat ran the the', 'the bill ran to the house the the'])
x_test = np.array(['the dog ran', 'the beautiful ran', 'the bat ran', 'the bill ran to the house'])


# 1 Tokenize Strings
def clean_str(x_text):
    """
    Params:
        Pandas Series of text
    Returns:
        Tokenized Pandas Seris of text
    """
    x_text = x_text.str.replace(r"[^A-Za-z0-9(),!?\'\`]", " ")
    x_text = x_text.str.replace(r"\'s", " \'s")
    x_text = x_text.str.replace(r"\'ve", " \'ve")
    x_text = x_text.str.replace(r"n\'t", " n\'t")
    x_text = x_text.str.replace(r"\'re", " \'re")
    x_text = x_text.str.replace(r"\'d", " \'d")
    x_text = x_text.str.replace(r"\'ll", " \'ll")
    x_text = x_text.str.replace(r",", " , ")
    x_text = x_text.str.replace(r"!", " ! ")
    x_text = x_text.str.replace(r"\(", " \( ")
    x_text = x_text.str.replace(r"\)", " \) ")
    x_text = x_text.str.replace(r"\?", " \? ")
    x_text = x_text.str.replace(r"\s{2,}", " ")
    x_text = x_text.str.strip()
    x_text = x_text.str.lower()
    return x_text


# helper function for concatenating n_grams
def to_list(x):
    grams = []
    for i in x:
        tmp = ''.join(i)
        grams.append(tmp)
    return grams


# function to return Pandas Series of n_grams
def get_n_grams(text, n_gram_size):
    text = text.str.split()
    text = text.apply(lambda x: ngrams(x, n_gram_size))
    return text.apply(lambda x: to_list(x))


def get_grams(x_text):
    # get unigrams
    cleaned = clean_str(x_text)
    unigram = cleaned.str.split()
    # print(unigram)
    # get bigrams
    bigrams = get_n_grams(cleaned, 2)
    # print(bigrams)
    # get trigrams
    trigrams = get_n_grams(cleaned, 3)
    # print(trigrams)
    return [unigram, bigrams, trigrams]


def build_dictionaries(docs_grammed):
    vocab_sizes = [5, 4, 3]
    vocab_sets = []
    for doc, size in zip(docs_grammed, vocab_sizes):
        vocab = build_vocabs(doc)
        vocab_sets.append(build_dictionary(vocab, size))
    return vocab_sets


def build_vocabs(doc):
    '''
    Params:
        Pandas Series containing list of n_grams
    Returns:
        strings of n_grams
    '''
    words = doc.apply(lambda x: ' '.join(x))
    words = words.str.cat(sep=' ')
    words = words.split()
    return words


# Step 2: Build the dictionary
def build_dictionary(vocab, vocabulary_size):
    '''
    Params:
        vocab: tokenized list of all the vocab in the vocabulary.
        vocabularly_size: max number of terms to be extracted from vocab
    Returns:
        vocabulary with only the top
    '''
    # print(vocab)
    count = [['UNK', -1]]
    count.extend(collections.Counter(vocab).most_common(vocabulary_size - 1))
    s = set()
    for word, _ in count:
        s.add(word)
    return s


def check_oov(x, j):
    '''
    Params:
        helper function applied to pd series
    Returns:
        list of updated words with low frequency words replaced
    '''
    checked = []
    for i in x:
        if i in j:
            checked.append(i)
        else:
            checked.append('oov')
    return checked


def build_oov(x_text):
    docs_grammed = get_grams(x_text)
    vocab_sets = build_dictionaries(docs_grammed)
    mapped_oov = []
    for i, j in zip(docs_grammed, vocab_sets):
        oov = i.apply(lambda x: check_oov(x, j))
        mapped_oov.append(oov)
    return mapped_oov


def concate_n_grams(x_text):
    mapped_oov = build_oov(x_text)
    x = pd.concat(mapped_oov, axis=1)
    y = x.ix[:, 0]+x.ix[:, 1]+x.ix[:, 2]
    return y


final_text = concate_n_grams(x_text)
final_text = final_text.apply(lambda x: ' '.join(x))
max_document_length = 30
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x_train = np.array(list(vocab_processor.fit_transform(final_text)))
print(x_train)
