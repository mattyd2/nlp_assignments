import os
import pandas as pd
import numpy as np
import time


def create_merge_files():
    data_set_types = ['train', 'test']
    file_types = ['../data/merged_train_data.txt',
                   '../data/merged_test_data.txt']
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
            df_1['review_type'] = ' __label__neg '
            df_1['review_and_label'] = df_1.ix[:, 0] + df_1['review_type']
            df_to_concat.append(df_1['review_and_label'])
        elif '/pos/' in path:
            text = []
            for i in files:
                txt_file = open(i, 'r')
                text.append(txt_file.read())
                txt_file.close()
            df_2 = pd.DataFrame(text)
            df_2['review_type'] = ' __label__pos '
            df_2['review_and_label'] = df_2.ix[:, 0] + df_2['review_type']
            df_to_concat.append(df_2['review_and_label'])
    final = pd.concat(df_to_concat, axis=0)
    final.to_csv('../data/merged_' + data_set_type +
                 '_data.txt', index=False, sep='\n', encoding='utf-8')

if __name__ == '__main__':
    create_merge_files()