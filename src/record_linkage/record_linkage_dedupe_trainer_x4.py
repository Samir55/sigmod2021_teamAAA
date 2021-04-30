import pandas as pd
import numpy as np
import os
import time
import csv
import re
import logging
import optparse
import re
import spacy
import dedupe
import pickle
import copy
import json
from unidecode import unidecode
import recordlinkage as rl
from omegaconf import OmegaConf

from .clean_datasets_x4 import clean_products_dataset_rf, clean_products_dataset_dedupe


def get_features(x_dev, candidate_links, dedupe_features=None):
    # Create the comparator functions
    compare_cl = rl.Compare(n_jobs=-1)
    compare_cl.exact('brand', 'brand')
    compare_cl.exact('product_type', 'product_type')
    compare_cl.exact('size', 'size')
    compare_cl.string('name', 'name', method='qgram')
    compare_cl.string('name', 'name', method='damerau_levenshtein')
    compare_cl.string('name', 'name', method='levenshtein')
    compare_cl.string('name', 'name', method='jarowinkler')
    compare_cl.string('name', 'name', method='smith_waterman')
    compare_cl.string('name', 'name', method='lcs')
    # compare_cl.string('price', 'price')

    features = compare_cl.compute(candidate_links, x_dev)

    if dedupe_features is not None:
        features = [features, dedupe_features]

    return features


if __name__ == '__main__':
    # Lightgbm
    x4_org = pd.read_csv('../../data/sigmod/X4.csv')
    x4 = clean_products_dataset_rf(x4_org)

    indexer = rl.Index()
    indexer.add(rl.index.Block('product_type'))
    indexer.add(rl.index.Block('brand'))
    indexer.add(rl.index.Block('size'))
    candidate_links = indexer.index(x4)

    t = time.time()
    features = get_features(x4, candidate_links)
    print("Time taken to get the features using 8 cores: {} sec".format(time.time() - t))

    # Dedupe trainer
    x4_org = pd.read_csv('../../data/sigmod/X4.csv')
    x4 = clean_products_dataset_dedupe(x4_org)

    params = OmegaConf.create()
    params.dataset_type = 'products'
    params.train_dataset_path = "../../data/sigmod/X4.csv"
    params.label_dataset_path = "../../data/sigmod/Y4.csv"
    params.save_model_path = "trained_x4.json"
    params.save_model_setting_path = "trained_x4_settings.json"
    params.columns = ['name', 'brand', 'size', 'product_type']
    params.training_file = 'tmp_products_train_data.json'
    params.sample_size = 1500
    params.recall = 0.9
    params.blocked_proportion = 0.9
    params.num_cores = 14
    params.index_predicates = True

    to_dedupe_dict = x4.to_dict(orient='index')
    fields = [
        {'field': 'name', 'type': 'Text', 'has missing': False},
        {'field': 'brand', 'type': 'Exact', 'has missing': False},
        {'field': 'size', 'type': 'Exact', 'has missing': False},
        {'field': 'product_type', 'type': 'Exact', 'has missing': False}]

    # Create deduper model
    print("Creating dedupe model.")
    deduper = dedupe.Dedupe(fields, num_cores=params.num_cores)

    # Get the label data
    y = pd.read_csv(params.label_dataset_path)

    trainig_data = {'match': [], 'distinct': []}
    match = y[y.label == 1].to_dict(orient='row')
    distinct = y[y.label == 0].to_dict(orient='row')
    for m in match:
        trainig_data['match'].append((to_dedupe_dict[m['left_instance_id']], to_dedupe_dict[m['right_instance_id']]))
    for d in distinct:
        trainig_data['distinct'].append((to_dedupe_dict[d['left_instance_id']], to_dedupe_dict[d['right_instance_id']]))

    # Save the training data
    with open(params.training_file, 'w') as fout:
        json.dump(trainig_data, fout)

    print("Preparing the training data.")
    with open(params.training_file) as tf:
        deduper.prepare_training(to_dedupe_dict, training_file=tf,
                                 sample_size=params.sample_size,
                                 blocked_proportion=params.blocked_proportion)

    # Train the model
    print("Training the model.")
    deduper.train(recall=params.recall, index_predicates=params.index_predicates)

    # Save the trained model
    print("Saving the model.")
    with open(params.save_model_path, 'w') as tf:
        deduper.write_training(tf)
    with open(params.save_model_setting_path, 'wb') as sf:
        deduper.write_settings(sf)

    print("Model predicates:")
    print(deduper.predicates)

    # Train
    # Cluster (prediction stage)
    # Please see why the accuracy dropped which cleaning has changed
    clustered_dupes = deduper.partition(to_dedupe_dict, 0.37)
    print('# duplicate sets', len(clustered_dupes))

    features['xy_same_entity'] = pd.Series(np.zeros(len(features)))
    features.xy_same_entity = 0.0

    # Save the result
    for el in clustered_dupes:
        for i in range(len(el[0])):
            for j in range(i + 1, len(el[0])):
                k = (el[0][i], el[0][j])
                r_k = (el[0][j], el[0][i])
                p = el[1][i] * el[1][j]

                if k in features.index:
                    features.loc[k, 'xy_same_entity'] = p

                if r_k in features.index:
                    features.loc[r_k, 'xy_same_entity'] = p

    # Create the labels
    gt_lbs = pd.read_csv(params.label_dataset_path)

    ls = np.zeros(len(features))
    labels = features.copy(deep=True)
    labels['label'] = ls

    len(gt_lbs)

    match = []
    mismatch = []

    for i in range(len(gt_lbs)):
        r = gt_lbs.iloc[i]
        if r.label == 1:
            match.append((r.left_instance_id, r.right_instance_id))

        else:
            mismatch.append((r.left_instance_id, r.right_instance_id))

    counted = 0
    true_inds = []

    for el in match:
        counted += (el in features.index)
        counted += ((el[1], el[0]) in features.index)

        if (el in features.index):
            true_inds.append(el)
        if ((el[1], el[0]) in features.index):
            true_inds.append((el[1], el[0]))

    print(counted)
    print(len(true_inds))

    counted = 0
    false_inds = []

    for el in mismatch:
        counted += (el in features.index)
        counted += ((el[1], el[0]) in features.index)

        if (el in features.index):
            false_inds.append(el)
        if ((el[1], el[0]) in features.index):
            false_inds.append((el[1], el[0]))

    print(counted)
    print(len(false_inds))

    labels.loc[true_inds, 'label'] = 1
    labels = labels['label']

    # Lib lightgbm
    import lightgbm as lgb


    def f1_metric(preds, train_data):

        labels = train_data.get_label()

        return 'f1', f1_score(labels, preds, average='weighted'), True


    from sklearn.metrics import f1_score


    def lgb_f1_score(y_hat, data):
        y_true = data.get_label()
        y_hat = np.round(y_hat)  # scikits f1 doesn't like probabilities
        return 'f1', f1_score(y_true, y_hat), True


    df_train, labels = features, labels

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        #     'metric': {'binary_logloss'},
        #     'num_leaves': 96,
        #     'max_depth': 10,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.95,
        'bagging_freq': 5
    }
    ROUNDS = 200
    evals_result = {}

    # Features to be trained on.
    # features_to_train_on = [0, 1, 2, 3, 4, 5, 6, 7, 8, 'xy_same_entity']

    # Preparing the input for the LightGBM model.
    d_train = lgb.Dataset(df_train,
                          label=labels)

    bst = lgb.train(params, d_train, ROUNDS, feval=lgb_f1_score, evals_result=evals_result)

    print("Training F1 score")
    print(lgb_f1_score(bst.predict(df_train), d_train))  # Non dedupe

    # Save the lightgbm model
    bst.save_model('x4_lgb_classifier.txt', num_iteration=bst.best_iteration)
