import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from data_provider.load_dataset import load_dataset
from data_provider.build_dataset import WeatherPredictionDataset, create_data_loaders
from omegaconf import OmegaConf
import warnings
import pdb

warnings.filterwarnings('ignore')

def create_dataloader(
                datadir: str,
                scale: str,
                batch_size: int,
                shuffle: bool,
                num_workers: int,
                pin_memory: bool,
                pre_training: bool,
                drop_last: bool,
                category_name: str,
                input_structure: str,
                data_version: str,
                lead_time_align: bool = False,
                split_rate: list = [0.7, 0.1, 0.2],
                align_time: bool = True,
                ):

    assert scale in (None, 'minmax', 'minmax_square', 'standard')
    assert category_name in ('solar', 'wind')
    assert input_structure in ('independence', 'merge')

    data_dict, ground_truth, features, scaler_dict = load_dataset(
                            datadir                = datadir,
                            category_name          = category_name,
                            scale                  = scale,
                            input_structure        = input_structure,
                            pre_training           = pre_training,
                            lead_time_align        = lead_time_align,
                            del_feature            = ['SWDDIR'],
                            data_version           = data_version,
                            split_rate             = split_rate,
                            align_time             = align_time,
                            )
    
    trn_dataset = WeatherPredictionDataset(data_dict        = data_dict,
                                            ground_truth    = ground_truth,
                                            category_name   = category_name,
                                            input_structure = input_structure,
                                            type            = 'train')
    val_dataset = WeatherPredictionDataset(data_dict        = data_dict,
                                            ground_truth    = ground_truth,
                                            category_name   = category_name,
                                            input_structure = input_structure,
                                            type            = 'val')
    tst_dataset = WeatherPredictionDataset(data_dict        = data_dict,
                                            ground_truth    = ground_truth,
                                            category_name   = category_name,
                                            input_structure = input_structure,
                                            type            = 'test')
    
    # torch dataloader
    trn_dataloader = create_data_loaders(trn_dataset,
                                            batch_size     =  batch_size,
                                            shuffle        =  shuffle,
                                            num_workers    =  num_workers,
                                            pin_memory     =  pin_memory,
                                            drop_last      =  drop_last)

    val_dataloader = create_data_loaders(val_dataset,
                                            batch_size     =  batch_size,
                                            shuffle        =  shuffle,
                                            num_workers    =  num_workers,
                                            pin_memory     =  pin_memory,
                                            drop_last      =  drop_last)

    tst_dataloader = create_data_loaders(tst_dataset,
                                            batch_size     =  batch_size,
                                            shuffle        =  False,
                                            num_workers    =  num_workers,
                                            pin_memory     =  pin_memory,
                                            drop_last      =  False)

    return trn_dataloader, val_dataloader, tst_dataloader, features, scaler_dict
