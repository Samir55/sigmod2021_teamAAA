# coding=utf-8
import deepmatcher as dm
import pandas as pd
import torch


def prepare_dataset():
    train, validation, test = dm.data.process(
        path='data/sigmod/',
        train='x2_train.csv',
        validation='x2_train.csv',
        test='x2_train.csv',
        ignore_columns=['left_instance_id', 'right_instance_id'],
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


def create_model():
    model = dm.MatchingModel(attr_summarizer='hybrid')
    return model


def train(model, datasets):
    model.run_train(
        datasets['train'],
        datasets['val'],
        epochs=20,
        batch_size=128,
        best_save_path='hybrid_model.pth',
        pos_neg_ratio=3)


if __name__ == '__main__':
    # Read the datasets
    datasets = prepare_dataset()

    # Create the model
    model = create_model()

    # Train the model
    train(model, datasets)

