import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class WeatherPredictionDataset(Dataset):
    def __init__(self, data_dict: dict, 
                 category_name: str, 
                 input_structure: str = 'independence', 
                 type: str = 'train', 
                 seq_len: int = 12, 
                 pred_len: int = 1):
        """
        시계열 데이터 예측을 위한 Dataset 클래스
        Args:
            data_dict (dict): 전처리된 데이터 딕셔너리
            category_name (str): 'solar' 또는 'wind'
            input_structure (str): 'independence' 또는 'merge'
            type (str): 'train', 'val', 'test'
            seq_len (int): 입력 데이터 길이 (Sliding Window 크기)
            pred_len (int): 예측할 타겟 길이 (Prediction Horizon)
        """
        # 데이터셋 저장 구조 초기화
        self.datasets = {'ECMWF': {}, 'LDAPS': {}}
        self.seq_len = seq_len
        self.pred_len = pred_len

        if input_structure == 'independence':
            for model_key in ['ECMWF', 'LDAPS']:
                if model_key not in data_dict:
                    continue
                
                for area_grp_id, entry in data_dict[model_key].items():
                    area_grp_id = str(area_grp_id)

                    # 입력 데이터와 타겟 데이터 매칭
                    if f"{type}_x" in entry and f"{type}_y" in entry:
                        inputs = entry[f"{type}_x"].copy()
                        targets = entry[f"{type}_y"].copy()

                        if area_grp_id not in self.datasets[model_key]:
                            self.datasets[model_key][area_grp_id] = {
                                'inputs': inputs,
                                'targets': targets,
                                'valid_idxs': []
                            }

                        current_dataset = self.datasets[model_key][area_grp_id]

                        # Windowing을 고려한 valid_idxs 생성
                        num_samples = len(inputs) - self.seq_len - self.pred_len + 1
                        if num_samples > 0:
                            current_dataset['valid_idxs'] = list(range(num_samples))

        elif input_structure == 'merge':
            if 'merge' in data_dict:
                for area_grp_id, entry in data_dict['merge'].items():
                    area_grp_id = str(area_grp_id)

                    # 입력 데이터와 타깃 데이터 매칭
                    if f"{type}_x" in entry and f"{type}_y" in entry:
                        inputs = entry[f"{type}_x"].copy()
                        targets = entry[f"{type}_y"].copy()

                        if area_grp_id not in self.datasets['ECMWF']:
                            self.datasets['ECMWF'][area_grp_id] = {
                                'inputs': inputs,
                                'targets': targets,
                                'valid_idxs': []
                            }

                        current_dataset = self.datasets['ECMWF'][area_grp_id]

                        # Windowing을 고려한 valid_idxs 생성
                        num_samples = len(inputs) - self.seq_len - self.pred_len + 1
                        if num_samples > 0:
                            current_dataset['valid_idxs'] = list(range(num_samples))

        else:
            raise ValueError("input_structure must be either 'independence' or 'merge'")

    def __getitem__(self, idx, model_key=None, area_grp_id=None):
        """
        시계열 데이터를 Sliding Window 방식으로 반환

        Args:
            idx (int): 데이터 인덱스
            model_key (str): 'ECMWF' 또는 'LDAPS'
            area_grp_id (str): 발전소 ID

        Returns:
            dict: 입력 데이터 (X), 타겟 데이터 (y)
        """
        if model_key is None or area_grp_id is None:
            raise ValueError("Both 'model_key' and 'area_grp_id' must be provided")

        dataset = self.datasets[model_key][area_grp_id]
        start_idx = dataset['valid_idxs'][idx]

        # 입력 시퀀스 생성 (seq_len 길이만큼)
        x_seq = dataset['inputs'].iloc[start_idx : start_idx + self.seq_len].values.astype(np.float32)

        # 예측할 타겟 시퀀스 생성 (pred_len 이후 값)
        y_seq = dataset['targets']['QGEN'].iloc[start_idx + self.seq_len : start_idx + self.seq_len + self.pred_len].values.astype(np.float32)

        return {
            'inputs': torch.tensor(x_seq, dtype=torch.float32),
            'targets': torch.tensor(y_seq, dtype=torch.float32)
        }

    def __len__(self, model_key=None, area_grp_id=None):
        """
        데이터 길이를 반환
        """
        if model_key is None or area_grp_id is None:
            total_length = sum(len(area['valid_idxs']) for model in self.datasets.values() for area in model.values())
            return total_length

        return len(self.datasets[model_key][area_grp_id]['valid_idxs'])

class FilteredDataset(Dataset):
    def __init__(self, dataset, model_key, area_grp_id):
        """
        특정 모델과 발전소 ID에 대한 필터링된 데이터셋
        """
        self.dataset = dataset
        self.model_key = model_key
        self.area_grp_id = area_grp_id

    def __len__(self):
        return self.dataset.__len__(model_key=self.model_key, area_grp_id=self.area_grp_id)

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx, model_key=self.model_key, area_grp_id=self.area_grp_id)

def create_data_loaders(dataset, 
                        batch_size, 
                        shuffle, 
                        num_workers, 
                        pin_memory, 
                        drop_last):
    """
    DataLoader 생성

    Args:
        dataset: WeatherPredictionDataset 객체
        batch_size: 배치 크기
        shuffle: 데이터 셔플 여부
        num_workers: 데이터 로딩 워커 수
        pin_memory: GPU 메모리 고정 여부
        drop_last: 마지막 배치를 버릴지 여부

    Returns:
        dict: 모델 및 발전소 ID별 DataLoader
    """
    data_loaders = {}

    for model_key, area_groups in dataset.datasets.items():
        for area_grp_id in area_groups.keys():
            filtered_dataset = FilteredDataset(dataset, model_key, area_grp_id)
            data_loaders[(model_key, area_grp_id)] = DataLoader(filtered_dataset,
                                                                batch_size=batch_size,
                                                                shuffle=shuffle,
                                                                num_workers=num_workers,
                                                                pin_memory=pin_memory,
                                                                drop_last=drop_last)
    return data_loaders