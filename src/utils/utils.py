import numpy as np
import torch
from tqdm import tqdm
import time
import logging
import os
import glob
import json
import shutil
import random

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    true = np.where(true == 0, 1e-5, true)  # Avoid division by zero
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    true = np.where(true == 0, 1e-5, true)  # Avoid division by zero
    return np.mean(np.square((pred - true) / true))

def NMAE(pred, true, region):
    capacity_dict = {
        '6498': 0.496,
        '5375': 0.49,
        '4593': 30.0,
        '4594': 30.0,
        '8711': 15.0
    }

    capacity = capacity_dict.get(region, None)

    return np.mean(np.abs(pred - true) / capacity) * 100

def cal_metric(pred, true, region=None):
    capacity_dict = {
        '6498': 0.496,
        '5375': 0.49,
        '4593': 30.0,
        '4594': 30.0,
        '8711': 15.0
    }
    
    # 설비용량 10% 이상인 데이터만 평가
    if region in capacity_dict:
        capacity = capacity_dict[region]
        valid_mask = true >= (capacity * 0.1)
        pred = pred[valid_mask]
        true = true[valid_mask]

    if len(true) == 0:
        print(f"[⚠️ Warning] No valid data points above 10% capacity for region: {region}")
        return {k: float('nan') for k in ['MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE', 'RSE', 'NMAE']}

    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    nmae = NMAE(pred, true, region)

    metric_dict = {
        'NMAE': round(nmae, 4),
        'MAE': round(mae, 4),
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'MAPE': round(mape, 4),
        'MSPE': round(mspe, 4),
        'RSE': round(rse, 4)
    }

    print(f'nmae: {metric_dict["NMAE"]:.4f}\t mse: {metric_dict["MSE"]:.4f}\t mae: {metric_dict["MAE"]:.4f}\t rmse: {metric_dict["RMSE"]:.4f}')
    return metric_dict

def version_build(accelerator, logdir: str, resume: int=None) -> str:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if not os.path.isdir(logdir):
            os.makedirs(logdir)
    if resume is None:
        version = len(os.listdir(logdir))
        load_dir = os.path.join(logdir, f'version{version}')
        save_dir = os.path.join(logdir, f'version{version}')
    else:
        version = resume
        load_dir = os.path.join(logdir, f'version{version}')
        save_dir = os.path.join(logdir, f'version{len(os.listdir(logdir))}')
        
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(save_dir)
    return load_dir, save_dir

def load_resume_model(model, loaddir: str, resume: int, pre_training: bool, fine_tuning_method: str):
    new_weights = torch.load(os.path.join(loaddir,'best_model.pt'))
    
    if not pre_training:
        # List of weights to be removed
        weights_to_remove = [
            'TFTpart2.recon_layer.weight',
            'TFTpart2.recon_layer.bias',
            'TFTpart2.mlp.0.weight',
            'TFTpart2.mlp.0.bias'
        ]
        # Remove the specified weights
        for weight in weights_to_remove:
            if weight in new_weights:
                del new_weights[weight]
                
    # load weights
    model.load_state_dict(new_weights, strict=False)

    if not pre_training:
        if fine_tuning_method == 'full':
            for param in model.parameters():
                param.requires_grad = True
            print(f'Fine tuning method: Full')
        elif fine_tuning_method == 'linear_probing':
            train_params = ['final_layers']
            for name, param in model.named_parameters():
                if any(x in name for x in train_params):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            print(f'Fine tuning method: Linear probing')
    print('load model from (version {})'.format(resume))

def prepare_data(current_loader, saved_data_dir="saved_data", status='train'):
    """
    Load or prepare data using model_key and area_grp_id extracted from current_loader.
    If data exists, load it; otherwise, process and save it.
    """
    dataset = current_loader.dataset  # Access the dataset object
    model_key = dataset.model_key  # Ensure the dataset has `model_key` attribute
    area_grp_id = dataset.area_grp_id  # Ensure the dataset has `area_grp_id` attribute
    
    # Define the directory structure
    model_dir = os.path.join(saved_data_dir, str(model_key))
    area_dir = os.path.join(model_dir, str(area_grp_id))
    os.makedirs(area_dir, exist_ok=True)
    
    # Define file paths
    x_file_path = os.path.join(area_dir, f"{status}_x.npy")
    y_file_path = os.path.join(area_dir, f"{status}_y.npy")
    
    if os.path.exists(x_file_path) and os.path.exists(y_file_path):
        print(f"Loading preprocessed data for (model_key={model_key}, area_grp_id={area_grp_id})...")
        all_x = np.load(x_file_path)
        all_y = np.load(y_file_path)
    else:
        print(f"Processing data for (model_key={model_key}, area_grp_id={area_grp_id})...")
        all_x, all_y = [], []
        total_batches = len(current_loader)
        
        for batch_idx, batch in enumerate(current_loader, start=1):
            if batch_idx % 100 == 0 or batch_idx == total_batches:
                print(f"Processing... {batch_idx}/{total_batches} ({(batch_idx / total_batches) * 100:.2f}%)")
            x, y = batch['inputs'], batch['targets']
            all_x.append(x.detach().cpu().numpy())
            all_y.append(y.detach().cpu().numpy())
        
        all_x = np.concatenate(all_x, axis=0)
        all_y = np.concatenate(all_y, axis=0)
        
        print(f"Saving processed data for (model_key={model_key}, area_grp_id={area_grp_id})...")
        np.save(x_file_path, all_x)
        np.save(y_file_path, all_y)
    
    return all_x, all_y

class Float32Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)
