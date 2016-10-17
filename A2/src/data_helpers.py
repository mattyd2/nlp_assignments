import re
import os
import itertools
import pandas as pd
import numpy as np
import collections
from nltk import ngrams
from sklearn.cross_validation import train_test_split
from tensorflow.contrib import learn
import tensorflow as tf
import pickle


def get_prepared_data(max_sentence_length, nb_grams):
    # Load data
    print("Loading data...")
    # Load Data (doesn't include Test Data)
    x_train, y_train = load_data_and_labels('train', nb_grams=nb_grams)
    x_train = x_train.apply(lambda x: ' '.join(x))
    vocab_processor = learn.preprocessing.VocabularyProcessor(
        max_sentence_length)

    print('fit transform training data...')
    vocab_processor.fit(x_train)
    x_train = np.array(list(vocab_processor.transform(x_train)))

    # Split Complete Training Data Set into Train and Validation
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.10, random_state=42)

    print('Loading test data...')
    x_test, y_test = load_data_and_labels('test', nb_grams=nb_grams)
    x_test = x_test.apply(lambda x: ' '.join(x))
    print('Transforming test data...')
    x_test = np.array(list(vocab_processor.transform(x_test)))
    # print(x_test)

    return x_train, y_train, x_val, y_val, x_test, y_test, vocab_processor


def load_data_and_labels(data_set_type, nb_grams):
    if data_set_type == 'train':
        # loaded = merge_data(data_set_type)
        loaded = pd.read_csv('../data/merged_train_data.csv')
    if data_set_type == 'test':
        # loaded = merge_data(data_set_type)
        loaded = pd.read_csv('../data/merged_test_data.csv')
    x = loaded.ix[:, 1]
    x = concate_n_grams(x, nb_grams)
    y = loaded.ix[:, 2]
    y = build_labels(y)
    return x, y


def concate_n_grams(x_text, nb_grams):
    mapped_oov = build_oov(x_text, nb_grams)
    if nb_grams == 1:
        return mapped_oov[0]
    elif nb_grams == 2:
        merged = pd.concat(mapped_oov, axis=1)
        x_text = merged.ix[:, 0] + merged.ix[:, 1]
        return x_text
    elif nb_grams == 3:
        merged = pd.concat(mapped_oov, axis=1)
        x_text = merged.ix[:, 0] + merged.ix[:, 1] + merged.ix[:, 2]
        return x_text


def build_oov(x_text, nb_grams):
    docs_grammed = get_grams(x_text, nb_grams)
    vocab_sets = build_dictionaries(docs_grammed, nb_grams)
    mapped_oov = []
    for doc_grammed, vocab_set in zip(docs_grammed, vocab_sets):
        oov = doc_grammed.apply(lambda x: check_oov(x, vocab_set))
        mapped_oov.append(oov)
    return mapped_oov


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


def get_grams(x_text, nb_grams):
    grammed_docs = []
    cleaned = clean_str(x_text)
    for i in range(0, nb_grams):
        num_grams = i+1
        grammed_docs.append(get_n_grams(cleaned, num_grams))
    return grammed_docs


# function to return Pandas Series of n_grams
def get_n_grams(text, n_gram_size):
    text = text.str.split()
    text = text.apply(lambda x: ngrams(x, n_gram_size))
    return text.apply(lambda x: to_list(x))


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


def build_dictionaries(docs_grammed, nb_grams):
    # vocab_sizes = [10000, 5000, 2500]
    vocab_sizes = [10000, 5000, 2500]
    # nb_grams = nb_grams-1
    print('nb_grams', nb_grams)
    vocab_sizes = vocab_sizes[:nb_grams]
    print('vocab_sizes', vocab_sizes)
    vocab_sets = []
    for doc, size in zip(docs_grammed, vocab_sizes[:nb_grams]):
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


def build_labels(y):
    '''
    Params:
        binary labels for negative and postive code to [0,1]
    '''
    y_labels = []
    for i in y.iteritems():
        if i[1] == 'neg':
            y_labels.append([0, 1])
        elif i[1] == 'pos':
            y_labels.append([1, 0])
    return np.array(y_labels)


# helper function for concatenating n_grams
def to_list(x):
    grams = []
    for i in x:
        tmp = ''.join(i)
        grams.append(tmp)
    return grams


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def create_merge_files():
    data_set_types = ['train', 'test']
    file_types = ['../data/merged_train_data.csv',
                   '../data/merged_test_data.csv']
    for i, j in zip(file_types, data_set_types):
        if os.path.exists(i):
            print(i, ' already exists...')
        else:
            print(i, 'does NOT exist, creating now...')
            merge_data(j)


def merge_data(data_set_type):
    neg_path = '../data/aclImdb/' + data_set_type + '/neg/'
    pos_path = '../data/aclImdb/' + data_set_type + '/pos/'
    paths = [neg_path, pos_path]
    df_to_concat = []
    for path in paths:
        output = os.listdir(path)
        files = []
        for i in output:
            files.append(path + i)
        if '/neg/' in path:
            text = []
            for i in files:
                txt_file = open(i, 'r')
                text.append(txt_file.read())
                txt_file.close()
            df_1 = pd.DataFrame(text)
            df_1['review_type'] = 'neg'
            df_to_concat.append(df_1)
        elif '/pos/' in path:
            text = []
            for i in files:
                txt_file = open(i, 'r')
                text.append(txt_file.read())
                txt_file.close()
            df_2 = pd.DataFrame(text)
            df_2['review_type'] = 'pos'
            df_to_concat.append(df_2)
    final = pd.concat(df_to_concat, axis=0)
    final.to_csv('../data/merged_' + data_set_type +
                 '_data.csv', encoding='utf-8')
    # return final


def get_prefix(nb_grams):
    if nb_grams == 1:
        abbreviation = 'uni_'
    elif nb_grams == 2:
        abbreviation = 'bi_'
    elif nb_grams == 3:
        abbreviation = 'tri_'
    return abbreviation


def pickle_data(to_pickle, labels, nb_grams):
    abbrv = get_prefix(nb_grams)
    for data, label in zip(to_pickle, labels):
        pickle.dump(data, open('../data/'+abbrv+label+'.p', "wb"))


def get_pickled_data(labels, nb_grams):
    print("Getting PICKLES...")
    abbrv = get_prefix(nb_grams)
    data = []
    for label in labels:
        data.append(pickle.load(open('../data/'+abbrv+label+'.p', "rb")))
    return data[0], data[1], data[2], data[3], data[4], data[5], data[6]


# if __name__ == '__main__':
#     check_for_merge_file('../data/merged_train_data.csv')
    # merge_data('train')
    # merge_data('test')
