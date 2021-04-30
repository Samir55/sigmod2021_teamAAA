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


def formatNumber(num):
    num = float(num)
    if num % 1 == 0:
        return int(num)
    else:
        return num


def fill_nulls_with_none(df):
    """ Fills nulls in a dataframe with None.
        This is required for the Dedupe package to work properly.

        Input: - dataframe with nulls as NaN

        Output: - new dataframe with nulls as None
    """
    new_df = df.copy()
    for col in df.columns:
        new_df[col] = new_df[col].where(new_df[col].notnull(), None)
    return new_df


def fill_col_nulls_with_mean(df, col_name):
    """ Fills nulls in a dataframe with None.
        This is required for the Dedupe package to work properly.

        Input: - dataframe with nulls as NaN

        Output: - new dataframe with nulls as None
    """
    new_df = df.copy()
    new_df[col_name] = new_df[col_name].where(new_df[col_name].notnull(), None)
    return new_df


def convert_numbers_to_strings(df, cols_to_convert, remove_point_zero=True):
    """ Convert number types to strings in a dataframe.
        This is convoluted as need to keep NoneTypes as NoneTypes for what comes next!

        Inputs: - df -> dataframe to convert number types
                - cols_to_convert -> list of columns to convert
                - remove_point_zero -> bool to say whether you want '.0' removed from number

        Ouputs: - dataframe with converted number types
    """
    new_df = df.copy()
    for col in cols_to_convert:
        if remove_point_zero:
            new_df[col] = new_df[col].apply(lambda x: str(x).replace('.0', '') \
                if not isinstance(x, type(None)) else x)
        else:
            new_df[col] = new_df[col].apply(lambda x: str(x) \
                if not isinstance(x, type(None)) else x)
    return new_df


def clean_products_dataset_rf(x_org):
    spacy.cli.download("en_core_web_sm")

    x4_dev = convert_numbers_to_strings(x_org, ['price']).copy(deep=True)
    #     x4_dev = x_org.copy(deep=True)
    x4_dev.set_index('instance_id', inplace=True)

    def get_type(record):
        name = record['name'].lower()

        if pd.isna(record['size']):
            if 'tv' in name:
                return 'tv'
            return 'mobile'

        flash_keywords = ['usb', 'drive']
        memory_stick_keywords = ['card', 'stick', 'sd', 'microsd', 'hc', 'class', 'speicherkarte']  # Add variants here

        is_flash = False
        is_memory = False

        for w in flash_keywords:
            if w in name:
                is_flash = True
                break

        for w in memory_stick_keywords:
            if w in name:
                is_memory = True
                break

        if is_flash:
            return 'flash'

        if is_memory:
            return 'stick'

        return 'stick'

    with open('../../data/sigmod/translations_lookup_all.json') as fin:
        variants = json.load(fin)

    with open('../../data/sigmod/langs_dict.json') as fin:
        json.load(fin)

    # Alpha numeric
    irrelevant_regex = re.compile(r'[^a-z0-9,.\-\s]')
    multispace_regex = re.compile(r'\s\s+')  # Why it doesn't work
    x4_dev.replace({r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)

    for column in x4_dev.columns:
        if column in ['instance_id', 'price']:
            continue
        x4_dev[column] = x4_dev[column].str.lower().str.replace(irrelevant_regex, ' ').str.replace(multispace_regex,
                                                                                                   ' ')

    x4_dev['product_type'] = x4_dev.apply(get_type, axis=1)
    #     x4_dev.drop('price', inplace=True, axis=1)
    x4_dev['size'] = x4_dev['size'].str.lower().str.replace(' ', '')
    x4_dev['size'] = x4_dev['size'].where(x4_dev['size'].notnull(), 0)

    # Remove unwanted words from the name
    for i in range(len(x4_dev)):
        record = x4_dev.iloc[i]

        name = record['name']

        # remove unnecessary characters
        basic_punct = '-/\*_,:;/()®™'
        punct_to_space = str.maketrans(basic_punct, ' ' * len(basic_punct))  # map punctuation to space
        name = name.translate(punct_to_space)

        # remove brand
        name = name.replace(record['brand'], '')

        # remove size

        if record.product_type in ['flash', 'stick']:
            name = re.sub('\d\d\d\s?gb', '', name, 6)
            name = re.sub('\d\d\s?gb', '', name, 6)
            name = re.sub('\d\s?gb', '', name, 6)

        tokens = name.split(' ')
        for wd, wdtl in variants.items():
            while wd in tokens:
                tokens.remove(wd)
            for wdt in wdtl:
                while wdt in tokens:
                    tokens.remove(wdt)

        unneeded_words = ['mmoire', 'speicherkarte', 'flashgeheugenkaart', 'flash', 'stick', 'speed', 'high']
        for w in unneeded_words:
            while w in tokens:
                tokens.remove(w)
        x4_dev.iloc[i]['name'] = ' '.join(tokens)

    for column in x4_dev.columns:
        if column in ['instance_id', 'price']:
            continue
        x4_dev[column] = x4_dev[column].str.lower().str.replace(irrelevant_regex, ' ').str.replace(multispace_regex,
                                                                                                   ' ')

    return x4_dev


def clean_products_dataset_dedupe(x_org):
    spacy.cli.download("en_core_web_sm")

    x4_dev = convert_numbers_to_strings(x_org, ['price']).copy(deep=True)
    x4_dev.set_index('instance_id', inplace=True)

    def get_type(record):
        name = record['name'].lower()

        if pd.isna(record['size']):
            if 'tv' in name:
                return 'tv'
            return 'mobile'

        flash_keywords = ['usb', 'drive']
        memory_stick_keywords = ['card', 'stick', 'sd', 'microsd', 'hc', 'class', 'speicherkarte']  # Add variants here

        is_flash = False
        is_memory = False

        for w in flash_keywords:
            if w in name:
                is_flash = True
                break

        for w in memory_stick_keywords:
            if w in name:
                is_memory = True
                break

        if is_flash:
            return 'flash'

        if is_memory:
            return 'stick'

        return 'stick'

    with open('../../data/sigmod/translations_lookup_all.json') as fin:
        variants = json.load(fin)

    with open('../../data/sigmod/langs_dict.json') as fin:
        json.load(fin)

    # Alpha numeric
    irrelevant_regex = re.compile(r'[^a-z0-9,.\-\s]')
    multispace_regex = re.compile(r'\s\s+')  # Why it doesn't work
    x4_dev.replace({r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)

    for column in x4_dev.columns:
        if column == 'instance_id':
            continue
        x4_dev[column] = x4_dev[column].str.lower().str.replace(irrelevant_regex, ' ').str.replace(multispace_regex,
                                                                                                   ' ')

    x4_dev['product_type'] = x4_dev.apply(get_type, axis=1)
    x4_dev.drop('price', inplace=True, axis=1)
    x4_dev['size'] = x4_dev['size'].str.lower().str.replace(' ', '')
    x4_dev['size'] = x4_dev['size'].where(x4_dev['size'].notnull(), 0)

    # Remove unwanted words from the name
    for i in range(len(x4_dev)):
        record = x4_dev.iloc[i]

        name = record['name']

        # remove unnecessary characters
        basic_punct = '-/\*_,:;/()®™'
        punct_to_space = str.maketrans(basic_punct, ' ' * len(basic_punct))  # map punctuation to space
        name = name.translate(punct_to_space)

        # remove brand
        name = name.replace(record['brand'], '')

        # remove size

        if record.product_type in ['flash', 'stick']:
            name = re.sub('\d\d\d\s?gb', '', name, 6)
            name = re.sub('\d\d\s?gb', '', name, 6)
            name = re.sub('\d\s?gb', '', name, 6)

        tokens = name.split(' ')
        for wd, wdtl in variants.items():
            while wd in tokens:
                tokens.remove(wd)
            for wdt in wdtl:
                while wdt in tokens:
                    tokens.remove(wdt)

        unneeded_words = ['mmoire', 'speicherkarte', 'flashgeheugenkaart', 'flash', 'stick', 'speed', 'high']
        for w in unneeded_words:
            while w in tokens:
                tokens.remove(w)
        x4_dev.iloc[i]['name'] = ' '.join(tokens)

    for column in x4_dev.columns:
        if column == 'instance_id':
            continue
        x4_dev[column] = x4_dev[column].str.lower().str.replace(irrelevant_regex, ' ').str.replace(multispace_regex,
                                                                                                   ' ')

    return x4_dev


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
    # compare_cl.string('name', 'name', method='smith_waterman')
    # compare_cl.string('name', 'name', method='lcs')
    # compare_cl.string('price', 'price')

    features = compare_cl.compute(candidate_links, x_dev)

    if dedupe_features is not None:
        features = [features, dedupe_features]

    return features


if __name__ == '__main__':
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
    from omegaconf import OmegaConf

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

    # Create training data dict compatibale with deduper
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
