import time

import dedupe
import pandas as pd

from clean_datasets import clean_laptops_dataset, clean_products_dataset

LOCAL = True

partition_threshold = {
    'x2': 0.5,
    'x3': 0.5,
    'x4': 0.5,
}


def deduper_eval(dataset_type: str, dataset):
    # Create deduper model
    with open('../../trained_models/deduper/trained_{}_settings.json'.format(dataset_type), 'rb') as fin:
        deduper = dedupe.StaticDedupe(fin,num_cores=8)

    # Prepare the data
    if dataset_type in ['x2', 'x3']:
        cols = [
            'instance_id',
            'brand',
            'cpu_brand',
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
    x2 = clean_laptops_dataset(x2)
    print("Evaluating X2 dataset")
    output.append(deduper_eval('x2', x2))

    print("Cleaning X3 dataset")
    x3 = clean_laptops_dataset(x3)
    print("Evaluating X3 dataset")
    # output.append(deduper_eval('x3', x3))

    print("Cleaning X4 dataset")
    x4 = clean_products_dataset(x4)
    print("Evaluating X4 dataset")
    # output.append(deduper_eval('x4', x4))

    output.to_csv('output.csv', index=False)
    print("Total elapsed time: {}".format(time.time() - start_time))
