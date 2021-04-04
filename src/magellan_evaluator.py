import sys
import py_entitymatching as em
import pandas as pd
import os
import joblib
import sys

if __name__ == '__main__':
    # Get the arguments
    n = len(sys.argv)
    print("Total arguments passed:", n)

    # Arguments passed
    print("\nName of Python script:", sys.argv[0])

    print("\nArguments passed:", end=" ")
    for i in range(1, n):
        print(sys.argv[i], end=" ")

    test_csv_file = sys.argv[1]

    # Read the dataset / Clean / Save
    A = pd.read_csv(test_csv_file)
    A = A.fillna(-999)
    A.to_csv('X_cleaned.csv', index=False)

    # Reread the cleaned dataset
    A = em.read_csv_metadata('X_cleaned.csv', key='instance_id')
    B = em.read_csv_metadata('X_cleaned.csv', key='instance_id')

    print('Number of tuples in A: ' + str(len(A)))
    print('Number of tuples in B: ' + str(len(B)))
    print('Number of tuples in A X B (i.e the cartesian product): ' + str(len(A) * len(B)))

    # Start running blocking
    ob = em.OverlapBlocker()

    C = ob.block_tables(A, B, 'title', 'title',
                        l_output_attrs=['instance_id', 'brand', 'cpu_brand', 'cpu_model', 'cpu_type',
                                        'cpu_frequency', 'ram_capacity', 'ram_type', 'ram_frequency',
                                        'hdd_capacity', 'ssd_capacity', 'weight', 'dimensions', 'title'],
                        r_output_attrs=['instance_id', 'brand', 'cpu_brand', 'cpu_model', 'cpu_type',
                                        'cpu_frequency', 'ram_capacity', 'ram_type', 'ram_frequency',
                                        'hdd_capacity', 'ssd_capacity', 'weight', 'dimensions', 'title'],
                        overlap_size=1, show_progress=True, l_output_prefix='left_',
                        r_output_prefix='right_', )
    # Get features
    feature_table = em.get_features_for_matching(A, B, validate_inferred_attr_types=False)

    # Get the attributes to be projected while predicting
    attrs_from_table = ['left_brand',
                        'left_cpu_brand', 'left_cpu_model', 'left_cpu_type',
                        'left_cpu_frequency', 'left_ram_capacity', 'left_ram_type',
                        'left_ram_frequency', 'left_hdd_capacity', 'left_ssd_capacity',
                        'left_weight', 'left_dimensions', 'left_title', 'right_brand',
                        'right_cpu_brand', 'right_cpu_model', 'right_cpu_type',
                        'right_cpu_frequency', 'right_ram_capacity', 'right_ram_type',
                        'right_ram_frequency', 'right_hdd_capacity', 'right_ssd_capacity',
                        'right_weight', 'right_dimensions', 'right_title']
    attrs_to_be_excluded = []
    attrs_to_be_excluded.extend(['_id', 'left_instance_id', 'right_instance_id'])
    attrs_to_be_excluded.extend(attrs_from_table)

    # Convert the cancidate set to feature vectors using the feature table
    L = em.extract_feature_vecs(C, feature_table=feature_table,
                                attrs_before=attrs_from_table,
                                show_progress=True, n_jobs=-1)

    loaded_rf = joblib.load("trained_models/random_forest.joblib")

    # Predict the matches
    predictions = loaded_rf.predict(table=L, exclude_attrs=attrs_to_be_excluded,
                                    append=True, target_attr='predicted', inplace=False, )


    # Prepare the output
    def duplicates(x):
        return x['left_instance_id'] == x['right_instance_id']


    def prepare_sigmod_output(res):
        ret = res[res.predicted == 1]
        ret = ret[['left_instance_id', 'right_instance_id']]
        ret = ret[~ret.apply(duplicates, axis=1)]
        return ret.drop_duplicates()


    ret = prepare_sigmod_output(predictions)
    ret.to_csv('output.csv', index=False)
