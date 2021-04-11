import argparse
from omegaconf import OmegaConf
from src.dedupe.deduper_trainer import dedupe_train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", help="dataset type", required=True, type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    params = OmegaConf.create()

    params.dataset_type = 'laptops'
    params.train_dataset_path = "../../data/sigmod/X{}.csv".format(args.d)
    params.label_dataset_path = "../../data/sigmod/Y{}.csv".format(args.d)
    params.save_model_path = "trained_x{}.json".format(args.d)
    params.save_model_setting_path = "trained_x{}_settings.json".format(args.d)
    params.columns = [
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
    params.training_file = 'tmp_products_train_data.json'
    params.sample_size = 1500
    params.recall = 0.9
    params.blocked_proportion = 0.9
    params.index_predicates = False  # TODO make true

    dedupe_train(params)
