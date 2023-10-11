import pytorch_lightning as pl
from datasets.dataset_generic import Generic_MIL_Dataset
from utils import get_loader_no_sampler, get_not_shuffled_loader
import os

def get_dataset(config_dict):
    dataset = Generic_MIL_Dataset(csv_path = config_dict['data_arguments']['ground_truth_csv'],
                                data_dir= config_dict['data_arguments']['feature_dir'],
                                shuffle = config_dict['data_arguments']['shuffle_data'], 
                                seed = config_dict['hyperparams_arguments']['seed'], 
                                print_info = config_dict['data_arguments']['print_info'],
                                label_dict = config_dict['data_arguments']['label_dict'],
                                patient_strat=config_dict['data_arguments']['patient_strat'],
                                ignore=[])
    return dataset.return_splits(from_id=False, 
                csv_path=os.path.join(config_dict['data_arguments']['split_dir'], 'splits_0.csv'))

class PLDataModule(pl.LightningDataModule):
    def __init__(self, config_dict):
        self.train_dataset, self.val_dataset, self.test_dataset = get_dataset(config_dict)
        self.val_dataset = get_not_shuffled_loader(self.val_dataset)
        self.test_dataset = get_not_shuffled_loader(self.test_dataset)
        self._log_hyperparams = None

    def train_dataloader(self):
        return get_loader_no_sampler(self.train_dataset)

    def prepare_data_per_node(self):
        pass

    def val_dataloader(self):
        return self.val_dataset

    def test_dataloader(self):
        return self.test_dataset
    
    def predict_dataloader(self):
        train_dataset = get_not_shuffled_loader(self.train_dataset)
        return [train_dataset, self.val_dataset]
