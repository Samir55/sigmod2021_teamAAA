# coding=utf-8
import copy

import deepmatcher as dm
import dill
import pandas as pd
import six
import torch
from deepmatcher.data import MatchingDataset


def prepare_dataset_for_train():
    train, validation, test = dm.data.process(
        path='data/sigmod/',
        train='x2_train.csv',
        validation='x2_train.csv',
        test=None,
        unlabeled=None,
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


def test(model):
    unlabeled = dm.data.process_unlabeled(
        path='data/sigmod/x2_test.csv',
        trained_model=model
    )

    predictions = model.run_prediction(unlabeled)
    predictions.to_csv('results.csv')


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


def post_process():
    pass


if __name__ == '__main__':
    evaluate = True

    # Create the model
    model = create_model()

    if evaluate:
        state = torch.load("hybrid_model.pth", map_location=torch.device('cpu'), pickle_module=dill)
        for k, v in six.iteritems(state):
            if k != 'model':
                model._train_buffers.add(k)
                setattr(model, k, v)

        if hasattr(model, 'state_meta'):
            train_info = copy.copy(model.state_meta)

            # Handle metadata manually.
            # TODO (Sid): Make this cleaner.
            train_info.metadata = train_info.orig_metadata
            MatchingDataset.finalize_metadata(train_info)

            model.initialize(train_info, model.state_meta.init_batch)

        model.load_state_dict(state['model'])

        test(model)

    else:
        # Read the datasets
        datasets = prepare_dataset_for_train()

        # Train the model
        train(model, datasets)
