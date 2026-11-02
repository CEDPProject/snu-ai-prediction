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
                model_name: str,
                seq_len: int,
                pred_len: int,
                target: str = 'power',
                lead_time_align: bool = False,
                split_rate: list = [0.7, 0.1, 0.2],
                align_time: bool = True,
                del_feature: list = None,
                split_date: str = None,
                end_date: str = None,
                target_type: str = None,
                ):

    assert scale in (None, 'minmax', 'minmax_square', 'standard')
    assert category_name in ('solar', 'wind')
    assert input_structure in ('independence', 'merge')
    assert target == 'power', f"DATASET.target='{target}' is not supported; only 'power' (QGEN) is implemented"

    data_dict, features, scaler_dict = load_dataset(
                            datadir                = datadir,
                            category_name          = category_name,
                            target                 = target,
                            scale                  = scale,
                            input_structure        = input_structure,
                            pre_training           = pre_training,
                            lead_time_align        = lead_time_align,
                            del_feature            = del_feature,
                            data_version           = data_version,
                            split_rate             = split_rate,
                            align_time             = align_time,
                            split_date             = split_date,
                            end_date               = end_date,
                            model_name             = model_name
                            )
    
    trn_dataset = WeatherPredictionDataset(data_dict        = data_dict,
                                            category_name   = category_name,
                                            input_structure = input_structure,
                                            seq_len         = seq_len, 
                                            pred_len        = pred_len,
                                            target_type     = target_type,
                                            type            = 'train')
    
    val_dataset = WeatherPredictionDataset(data_dict        = data_dict,
                                            category_name   = category_name,
                                            input_structure = input_structure,
                                            seq_len         = seq_len, 
                                            pred_len        = pred_len,      
                                            target_type     = target_type,                                      
                                            type            = 'val')
    
    tst_dataset = WeatherPredictionDataset(data_dict        = data_dict,
                                            category_name   = category_name,
                                            input_structure = input_structure,
                                            seq_len         = seq_len, 
                                            pred_len        = pred_len,
                                            target_type     = target_type,
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
    
    if 'ML' in model_name:
        return data_dict, data_dict, data_dict, features, scaler_dict
    else:
        return trn_dataloader, val_dataloader, tst_dataloader, features, scaler_dict
