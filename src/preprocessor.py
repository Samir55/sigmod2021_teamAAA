# coding=utf-8
import deepmatcher as dm
import pandas as pd
import torch

import deepmatcher as dm


def prepare_dataset():
    train, validation, test = dm.data.process(
        path='data/sample_data/itunes-amazon',
        train='train.csv',
        validation='validation.csv',
        test='test.csv',
        ignore_columns=['left_id', 'right_id'],
        left_prefix='left_',
        right_prefix='right_',
        label_attr='label',
        id_attr='id',
        tokenize='nltk',
        lowercase=False,
        embeddings='fasttext.en.bin')

    return {
        'train': train,
        'val': validation,
        'test': test
    }
