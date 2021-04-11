import json

import dedupe
import pandas as pd
from omegaconf import OmegaConf, DictConfig

from src.dedupe.clean_datasets import clean_laptops_dataset, clean_products_dataset, formatNumber


def dedupe_train(params: DictConfig):
    # Print parameters
    print("Parameters used")
    print(params)

    # Read the dataset
    print("Reading the datasets.")
    x_org = pd.read_csv(params.train_dataset_path)

    if params.dataset_type == 'laptops':
        x_dev = clean_laptops_dataset(x_org)
    else:
        x_dev = clean_products_dataset(x_org)

    # Create training data dict compatibale with deduper
    to_dedupe = x_dev[params.columns].copy()
    to_dedupe_dict = to_dedupe.to_dict(orient='index')

    # Get the fields
    if params.dataset_type == 'laptops':
        screen_sizes = set(pd.read_csv('../../data/sigmod/laptops.csv').Inches)
        screen_sizes = [str(formatNumber(str(s).lower())) for s in screen_sizes]
        extra_brands = set(pd.read_csv('../../data/sigmod/laptops.csv').Company.str.lower().unique())

        fields = [{'field': 'brand', 'type': 'Exact', 'has_missing': True},
                  {'field': 'cpu_brand', 'type': 'Exact', 'has_missing': True},
                  # {'field' : 'cpu_model', 'type': 'String', 'has_missing' : True},
                  {'field': 'cpu_type', 'type': 'Exact', 'has_missing': True},
                  {'field': 'ram_capacity', 'type': 'Exact', 'has_missing': True},
                  {'field': 'hdd_capacity', 'type': 'Exact', 'has_missing': True},
                  {'field': 'ssd_capacity', 'type': 'Exact', 'has_missing': True},
                  {'field': 'title', 'type': 'Text', 'has_missing': True},
                  {'field': 'screen_size', 'type': 'Categorical', 'has_missing': True, 'categories': screen_sizes},
                  {'field': 'model', 'type': 'String', 'has_missing': True}]
    else:
        fields = [
            {'field': 'name', 'type': 'String', 'has_missing': False},
            {'field': 'brand',
             'type': 'Categorical', 'categories': list(x_dev.brand.unique()),
             'has_missing': False},
            {'field': 'size', 'type': 'Exact', 'has_missing': False},
            {'field': 'product_type',
             'type': 'Categorical', 'categories': list(x_dev.product_type.unique()),
             'has_missing': False}]

    # Create deduper model
    print("Creating dedupe model.")
    deduper = dedupe.Dedupe(fields, num_cores=14)

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
