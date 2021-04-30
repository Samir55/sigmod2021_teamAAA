import time

import dedupe
import pandas as pd
import recordlinkage as rl
import lightgbm as lgb
import numpy as np

from clean_datasets_2 import clean_laptops_dataset as clean_x2
from clean_datasets_3 import clean_laptops_dataset as clean_x3
from record_linkage_dedupe_trainer_x4 import clean_products_dataset_dedupe as clean_x4_dedupe
from record_linkage_dedupe_trainer_x4 import clean_products_dataset_rf as clean_x4_rf

LOCAL = True
NUM_CORES = 8

partition_threshold = {
    'x2': 0.3,
    'x3': 0.3,
    'x4': 0.37,
}


def deduper_eval(dataset_type: str, dataset):
    # Create deduper model
    with open('../../trained_models/deduper/sub_5/trained_{}_settings.json'.format(dataset_type), 'rb') as fin:
        deduper = dedupe.StaticDedupe(fin, num_cores=8)

    # Prepare the data
    if dataset_type in ['x2', 'x3']:
        cols = [
            'instance_id',
            'brand',
            'model_name',
            # 'model_number',
            'cpu_brand',
            'cpu_model',
            'cpu_type',
            'ram_capacity',
            'hdd_capacity',
            'ssd_capacity',
            'title',
            'screen_size',
            'model']
    else:
        cols = ['name', 'brand', 'size', 'product_type']
    to_dedupe = dataset[cols]
    to_dedupe_dict = to_dedupe.to_dict(orient='index')

    # Cluster (prediction stage)
    clustered_dupes = deduper.partition(to_dedupe_dict, partition_threshold[dataset_type])
    print('# duplicate sets', len(clustered_dupes))

    # Save the result
    res = []
    for el in clustered_dupes:
        for i in range(len(el[0])):
            for j in range(i + 1, len(el[0])):
                res.append((el[0][i], el[0][j]))

    res_df = pd.DataFrame(res)
    res_df.columns = ['left_instance_id', 'right_instance_id']
    return res_df


def eval_lightgbm_dedupe(dataset_type: str, dataset_rf, dataset_dedupe):
    # Create deduper model
    with open('../../trained_models/combined_2/trained_{}_settings.json'.format(dataset_type), 'rb') as fin:
        deduper = dedupe.StaticDedupe(fin, num_cores=NUM_CORES)

    cols = ['name', 'brand', 'size', 'product_type']

    to_dedupe = dataset_dedupe[cols]
    to_dedupe_dict = to_dedupe.to_dict(orient='index')

    # Cluster (prediction stage)
    clustered_dupes = deduper.partition(to_dedupe_dict, partition_threshold[dataset_type])
    print('# duplicate sets', len(clustered_dupes))

    # Create the record linkage model
    # Indexer
    indexer = rl.Index()
    indexer.add(rl.index.Block('product_type'))
    indexer.add(rl.index.Block('brand'))
    indexer.add(rl.index.Block('size'))
    candidate_links = indexer.index(dataset_rf)

    # Comparing
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

    # Features
    features = compare_cl.compute(candidate_links, dataset_rf)

    # Add dedupe features
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

    # Now load the lightgbm
    bst = lgb.Booster(model_file='../../trained_models/combined_2/x4_lgb_classifier.txt')

    # Predict
    confs = bst.predict(features)
    features['label'] = confs

    # Save the csv file
    # Now export the left and right instance ids
    # Save the result
    res = []
    for i in range(len(confs)):
        record = features.iloc[i]
        label = record.label
        if label > 0.5:
            res.append(record.name)

    res_df = pd.DataFrame(res)
    res_df.columns = ['left_instance_id', 'right_instance_id']

    return res_df


if __name__ == '__main__':
    start_time = time.time()
    print("Start")
    # Read the datasets
    s_x2 = pd.read_csv('../../data/sigmod/X2.csv')
    s_x3 = pd.read_csv('../../data/sigmod/X3.csv')
    s_x4 = pd.read_csv('../../data/sigmod/X4.csv')

    #
    # Reverse the shuffling effect
    #
    # Detect which one is x4
    rem = []
    if len(s_x2.columns) == 5:
        x4 = s_x2
        rem.extend([s_x3, s_x4])
    elif len(s_x3.columns) == 5:
        x4 = s_x3
        rem.extend([s_x2, s_x4])
    else:
        x4 = s_x4
        rem.extend([s_x2, s_x3])

    # Determine x2 and x3
    output = pd.DataFrame(columns=['left_instance_id', 'right_instance_id'])
    if not LOCAL:
        if len(rem[0]) > len(rem[1]):
            x3 = rem[0]
            x2 = rem[1]
        else:
            x3 = rem[1]
            x2 = rem[0]
    else:
        x2 = s_x2
        x3 = s_x3
        x4 = s_x4

    # Now, we evaluate based on the trained models
    print("Cleaning X2 dataset")
    x2 = clean_x2(x2)
    print("Evaluating X2 dataset")
    output = output.append(deduper_eval('x2', x2))

    print("Cleaning X3 dataset")
    x3 = clean_x3(x3)
    print("Evaluating X3 dataset")
    output = output.append(deduper_eval('x3', x3))

    print("Cleaning X4 dataset")
    x4_rf = clean_x4_rf(x4)
    x4_dedupe = clean_x4_dedupe(x4)

    # Split by brand, size, type
    print("Evaluating X4 dataset")
    output = output.append(eval_lightgbm_dedupe('x4', dataset_rf=x4_rf, dataset_dedupe=x4_dedupe))

    output.to_csv('output.csv', index=False)
    print("Total elapsed time: {}".format(time.time() - start_time))
