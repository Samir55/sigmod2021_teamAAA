from omegaconf import OmegaConf
from deduper_trainer import dedupe_train

if __name__ == '__main__':
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


    dedupe_train(params)
