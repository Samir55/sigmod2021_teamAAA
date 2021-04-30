import csv
import json
import logging
import dedupe
import pandas as pd
from omegaconf import OmegaConf, DictConfig
from itertools import permutations, combinations
from tqdm import tqdm
from clean_datasets_2 import clean_laptops_dataset as clean_x2
from clean_datasets_3_new import clean_laptops_dataset as clean_x3
from clean_datasets_4 import clean_products_dataset as clean_x4
from clean_datasets_4 import formatNumber


# TODO remove multispace, remove touch word, add core i5 640m and core i7 640m
def dedupe_train(params: DictConfig):
    logging.getLogger().setLevel(logging.DEBUG)

    # Print parameters
    print("Parameters used")
    print(params)

    # Read the dataset
    print("Reading the datasets.")
    x_org = pd.read_csv(params.train_dataset_path)

    if params.dataset_type == 'laptops_2':
        x_dev = clean_x2(x_org)
    elif params.dataset_type == 'laptops_3':
        x_dev = clean_x3(x_org)
    else:
        x_dev = clean_x4(x_org)

    x_dev.to_csv('cleaning.csv')
    # Create training data dict compatibale with deduper
    to_dedupe_dict = x_dev.to_dict(orient='index')

    # Get the fields
    if params.dataset_type == 'laptops_2':
        fields = [
            # {'field': 'brand', 'type': 'Exact', 'has missing': True},
            # {'field': 'cpu_brand', 'type': 'Exact', 'has missing': True},
            {'field': 'cpu_model', 'type': 'Exact', 'has missing': True},
            # {'field': 'model_name', 'type': 'Exact', 'has missing': True},
            {'field': 'cpu_type', 'type': 'Exact', 'has missing': True},
            {'field': 'ram_capacity', 'type': 'Exact', 'has missing': True},
            {'field': 'hdd_capacity', 'type': 'Exact', 'has missing': True},
            {'field': 'ssd_capacity', 'type': 'Exact', 'has missing': True},
            # {'field': 'new_title', 'type': 'Text', 'has missing': True},
            # {'field': 'screen_size', 'type': 'Exact', 'has missing': True},
            {'field': 'model', 'type': 'Text', 'has missing': True}
        ]
    elif params.dataset_type == 'laptops_3':
        screen_sizes = set(pd.read_csv('../../data/sigmod/laptops.csv', encoding='windows-1251').Inches)
        screen_sizes = [str(formatNumber(str(s).lower())) for s in screen_sizes]
        extra_brands = set(
            pd.read_csv('../../data/sigmod/laptops.csv', encoding='windows-1251').Company.str.lower().unique())

        fields = [
            {'field': 'brand', 'type': 'Exact', 'has missing': True},
            # {'field': 'cpu_brand', 'type': 'Exact', 'has missing': True},
            {'field': 'cpu_model', 'type': 'Text', 'has missing': True},
            # {'field': 'model_name', 'type': 'Exact', 'has missing': True},
            {'field': 'cpu_type', 'type': 'Exact', 'has missing': True},
            {'field': 'ram_capacity', 'type': 'Exact', 'has missing': True},
            # {'field': 'hdd_capacity', 'type': 'Exact', 'has missing': True},
            # {'field': 'ssd_capacity', 'type': 'Exact', 'has missing': True},
            {'field': 'new_title', 'type': 'Text', 'has missing': True},
            # {'field': 'screen_size', 'type': 'Exact', 'has missing': True},
            {'field': 'model', 'type': 'Text', 'has missing': True}
        ]
    else:
        fields = [
            {'field': 'name', 'type': 'Text', 'has missing': True},
            {'field': 'brand', 'type': 'Exact', 'has missing': True},
            {'field': 'size', 'type': 'Exact', 'has missing': True},
            {'field': 'product_type', 'type': 'Exact', 'has missing': True}]

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

    # deduper.active_learner()

    # Save the trained model
    print("Saving the model.")
    with open(params.save_model_path, 'w') as tf:
        deduper.write_training(tf)
    with open(params.save_model_setting_path, 'wb') as sf:
        deduper.write_settings(sf)

    print("Model predicates:")
    print(deduper.predicates)

    # Write our original data back out to a CSV with a new column called
    # 'Cluster ID' which indicates which records refer to each other.
    clustered_dupes = deduper.partition(to_dedupe_dict, 0.4)

    print('# duplicate sets', len(clustered_dupes))
    cluster_membership = {}
    result_list = []
    counter = 0
    for cluster_id, (records, scores) in enumerate(clustered_dupes):
        print(cluster_id, records, scores)
        for cb in combinations(records, 2):
            # print(cb)
            result_list.append({"left_instance_id": cb[0], "right_instance_id": cb[1]})
            counter += 1
        for record_id, score in zip(records, scores):
            cluster_membership[record_id] = {
                "Cluster ID": cluster_id,
                "confidence_score": score
            }

    from collections import defaultdict
    current_cluster_id = 0
    record_to_cluster = {}
    cluster_records = defaultdict(set)

    gold = pd.read_csv(params.label_dataset_path)
    gold_dict = {}
    lii = "left_instance_id"
    rii = "right_instance_id"
    lbl = "label"

    for index, row in tqdm(gold.iterrows()):
        l, r, label = row[lii], row[rii], row[lbl]
        if label == 0:
            continue

        s = l + "|" + r
        gold_dict[s] = 1

    for r in gold_dict:
        a, b = (r.split('|'))

        # Check the case
        if (a not in record_to_cluster and b not in record_to_cluster):
            # Create a new cluster
            record_to_cluster[a] = current_cluster_id
            record_to_cluster[b] = current_cluster_id
            cluster_records[current_cluster_id].update({a, b})
            current_cluster_id += 1
        elif (a in record_to_cluster and b not in record_to_cluster):
            c = record_to_cluster[a]
            record_to_cluster[b] = c
            cluster_records[c].update({b})
            pass
        elif (b in record_to_cluster and a not in record_to_cluster):
            c = record_to_cluster[b]
            record_to_cluster[a] = c
            cluster_records[c].update({a})
        elif (a in record_to_cluster and b in record_to_cluster and record_to_cluster[a] != record_to_cluster[b]):
            # merge two clusters
            # Get the cluster id of to be kept cluster
            c_keep = record_to_cluster[a]
            c_rem = record_to_cluster[b]
            record_to_cluster[b] = c_keep
            cluster_records[c_keep].update(cluster_records[c_rem])
            del cluster_records[c_rem]

    print("total:", counter)
    with open('debug.csv', 'w') as f_output, open(params.train_dataset_path) as f_input:

        reader = csv.DictReader(f_input)
        # get first element
        fieldnames = ['Cluster ID', 'confidence_score', 'Real Cluster ID'] + list(
            to_dedupe_dict[list(to_dedupe_dict.keys())[0]].keys())
        # print(fieldnames)

        writer = csv.DictWriter(f_output, fieldnames=fieldnames)
        writer.writeheader()

        for k, v in to_dedupe_dict.items():
            v.update(cluster_membership[k])
            # Real cluster id
            if v[k] not in record_to_cluster:
                record_to_cluster[v[k]] = current_cluster_id
                current_cluster_id += 1
            v.update(record_to_cluster[v[k]])
            # print(v.keys())
            writer.writerow(v)
