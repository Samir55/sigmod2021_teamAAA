# coding=utf-8
import deepmatcher as dm
import pandas as pd
import torch

# coding=utf-8
import deepmatcher as dm
import pandas as pd
import torch


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


def create_model():
    model = dm.MatchingModel(attr_summarizer='hybrid')

    return model


def train(model, datasets):
    model.run_train(
        datasets['train'],
        datasets['val'],
        epochs=10,
        batch_size=16,
        best_save_path='hybrid_model.pth',
        pos_neg_ratio=3)



if __name__ == '__main__':
    # Read the datasets
    datasets = prepare_dataset()

    # Create the model
    model = create_model()

    # Train the model
    train()
