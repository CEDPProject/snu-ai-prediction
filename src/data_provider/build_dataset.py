import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class WeatherPredictionDataset(Dataset):
    def __init__(self, data_dict: dict, ground_truth: pd.DataFrame,
                 category_name: str, input_structure: str = 'independence',
                 type: str = 'train'):

        # 데이터셋 저장 구조 초기화
        self.datasets = {'ECMWF': {}, 'LDAPS': {}}

        if input_structure == 'independence':
            for model_key in ['ECMWF', 'LDAPS']:
                for area_grp_id, entry in data_dict[model_key].items():
                    area_grp_id = str(area_grp_id)  # Ensure AREA_GRP_ID is a string
                    df = entry[type].copy()

                    # 병합 전에 데이터 유형 통일
                    df['AREA_GRP_ID'] = df['AREA_GRP_ID'].astype(str)
                    ground_truth['ID'] = ground_truth['ID'].astype(str)
                    df['FCST_TM'] = pd.to_datetime(df['FCST_TM'])
                    ground_truth['TM'] = pd.to_datetime(ground_truth['TM'])

                    # 'AREA_GRP_ID'를 기준으로 그룹화
                    location_groups = df.groupby('AREA_GRP_ID')

                    for local_area_grp_id, group_df in location_groups:
                        local_area_grp_id = str(local_area_grp_id)  # Ensure AREA_GRP_ID is a string
                        merged = pd.merge(
                            group_df,
                            ground_truth,
                            left_on=['AREA_GRP_ID', 'FCST_TM'],
                            right_on=['ID', 'TM'],
                            how='inner'
                        )

                        full_area_grp_id = f"{area_grp_id}"

                        if full_area_grp_id not in self.datasets[model_key]:
                            self.datasets[model_key][full_area_grp_id] = {
                                'inputs': pd.DataFrame(),
                                'targets': pd.Series(dtype='float64'),
                                'valid_idxs': []
                            }

                        # 데이터 추가
                        inputs = merged.drop(columns=['QGEN', 'ID', 'TM']).reset_index(drop=True)
                        targets = merged['QGEN'].reset_index(drop=True)

                        current_dataset = self.datasets[model_key][full_area_grp_id]
                        current_dataset['inputs'] = pd.concat([current_dataset['inputs'], inputs], ignore_index=True)
                        current_dataset['targets'] = pd.concat([current_dataset['targets'], targets], ignore_index=True)
                        current_dataset['valid_idxs'] = np.arange(len(current_dataset['inputs']))

        elif input_structure == 'merge':
            for area_grp_id, entry in data_dict['merge'].items():
                area_grp_id = str(area_grp_id)  # Ensure AREA_GRP_ID is a string
                df = entry[type].copy()
                
                # 병합 전에 데이터 유형 통일
                df['AREA_GRP_ID'] = df['AREA_GRP_ID'].astype(str)
                ground_truth['ID'] = ground_truth['ID'].astype(str)
                df['FCST_TM'] = pd.to_datetime(df['FCST_TM'])
                ground_truth['TM'] = pd.to_datetime(ground_truth['TM'])

                location_groups = df.groupby('AREA_GRP_ID')

                for local_area_grp_id, group_df in location_groups:
                    local_area_grp_id = str(local_area_grp_id)  # Ensure AREA_GRP_ID is a string
                    merged = pd.merge(
                        group_df,
                        ground_truth,
                        left_on=['AREA_GRP_ID', 'FCST_TM'],
                        right_on=['ID', 'TM'],
                        how='inner'
                    )

                    full_area_grp_id = f"{area_grp_id}"

                    if full_area_grp_id not in self.datasets['ECMWF']:
                        self.datasets['ECMWF'][full_area_grp_id] = {
                            'inputs': pd.DataFrame(),
                            'targets': pd.Series(dtype='float64'),
                            'valid_idxs': []
                        }

                    # 데이터 추가
                    inputs = merged.drop(columns=['QGEN', 'ID', 'TM']).reset_index(drop=True)
                    targets = merged['QGEN'].reset_index(drop=True)

                    current_dataset = self.datasets['ECMWF'][full_area_grp_id]
                    current_dataset['inputs'] = pd.concat([current_dataset['inputs'], inputs], ignore_index=True)
                    current_dataset['targets'] = pd.concat([current_dataset['targets'], targets], ignore_index=True)
                    current_dataset['valid_idxs'] = np.arange(len(current_dataset['inputs']))
        else:
            raise ValueError("input_structure must be either 'independence' or 'merge'")
        
    def __getitem__(self, idx, model_key=None, area_grp_id=None):
        if model_key is None or area_grp_id is None:
            raise ValueError("Both 'model_key' and 'area_grp_id' must be provided")

        model_key = str(model_key)
        area_grp_id = str(area_grp_id)

        dataset = self.datasets[model_key][area_grp_id]
        i = dataset['valid_idxs'][idx]

        # Extract input features
        exclude_columns = ['NWP_TP_CD', 'AREA_GRP_ID', 'CRTN_TM', 'FCST_TM', 'LEAD_TM']
        input_features = dataset['inputs'].drop(
            columns=[col for col in exclude_columns if col in dataset['inputs'].columns]
        ).iloc[i]
        input_features = np.array(input_features, dtype=np.float32)

        # Extract target value
        target_value = dataset['targets'].iloc[i]

        return {
            'inputs': torch.tensor(input_features, dtype=torch.float32),
            'targets': torch.tensor([target_value], dtype=torch.float32)
        }
    
    def __len__(self, model_key=None, area_grp_id=None):
        if model_key is None or area_grp_id is None:
            total_length = 0
            for mk, area_groups in self.datasets.items():
                for ag_id in area_groups.keys():
                    total_length += len(self.datasets[mk][ag_id]['valid_idxs'])
            return total_length

        model_key = str(model_key)
        area_grp_id = str(area_grp_id)
        return len(self.datasets[model_key][area_grp_id]['valid_idxs'])

class FilteredDataset(Dataset):
    def __init__(self, dataset, model_key, area_grp_id):
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
    data_loaders = {}

    for model_key, area_groups in dataset.datasets.items():
        for area_grp_id in area_groups.keys():
            # 필터링된 데이터셋 생성
            filtered_dataset = FilteredDataset(dataset, model_key, area_grp_id)
            # DataLoader 생성
            data_loaders[(model_key, area_grp_id)] = DataLoader(filtered_dataset,
                                                        batch_size     =  batch_size,
                                                        shuffle        =  shuffle,
                                                        num_workers    =  num_workers,
                                                        pin_memory     =  pin_memory,
                                                        drop_last      =  drop_last)
    return data_loaders

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class WeatherPredictionDataset(Dataset):
    def __init__(self, data_dict: dict,
                 category_name: str,
                 input_structure: str = 'independence',
                 type: str = 'train'):
        """
        WeatherPredictionDataset 생성자.

        Args:
            data_dict: 전처리된 데이터 딕셔너리 (data_dict['ECMWF']['6498']['train_x'] 등).
            category_name: 'solar' 또는 'wind'.
            input_structure: 'independence' 또는 'merge'.
            type: 'train', 'val', 'test'.
        """
        # 데이터셋 저장 구조 초기화
        self.datasets = {'ECMWF': {}, 'LDAPS': {}}

        if input_structure == 'independence':
            for model_key in ['ECMWF', 'LDAPS']:
                if model_key not in data_dict:
                    continue

                for area_grp_id, entry in data_dict[model_key].items():
                    area_grp_id = str(area_grp_id)  # Ensure AREA_GRP_ID is a string

                    # 입력 데이터와 타깃 데이터 매칭
                    if f"{type}_x" in entry and f"{type}_y" in entry:
                        inputs = entry[f"{type}_x"]
                        targets = entry[f"{type}_y"]

                        if area_grp_id not in self.datasets[model_key]:
                            self.datasets[model_key][area_grp_id] = {
                                'inputs': pd.DataFrame(),
                                'targets': pd.DataFrame(columns=['ID', 'QGEN']),
                                'valid_idxs': []
                            }
                        current_dataset = self.datasets[model_key][area_grp_id]
                        current_dataset['inputs'] = pd.concat([current_dataset['inputs'], inputs], ignore_index=True)
                        current_dataset['targets'] = pd.concat([current_dataset['targets'], targets], ignore_index=True)
                        current_dataset['valid_idxs'] = np.arange(len(current_dataset['inputs']))

        elif input_structure == 'merge':
            if 'merge' in data_dict:
                for area_grp_id, entry in data_dict['merge'].items():
                    area_grp_id = str(area_grp_id)

                    # 입력 데이터와 타깃 데이터 매칭
                    if f"{type}_x" in entry and f"{type}_y" in entry:
                        inputs = entry[f"{type}_x"]
                        targets = entry[f"{type}_y"]

                        if area_grp_id not in self.datasets['ECMWF']:
                            self.datasets['ECMWF'][area_grp_id] = {
                                'inputs': pd.DataFrame(),
                                'targets': pd.DataFrame(columns=['ID', 'QGEN']),
                                'valid_idxs': []
                            }

                        current_dataset = self.datasets['ECMWF'][area_grp_id]
                        current_dataset['inputs'] = pd.concat([current_dataset['inputs'], inputs], ignore_index=True)
                        current_dataset['targets'] = pd.concat([current_dataset['targets'], targets], ignore_index=True)
                        current_dataset['valid_idxs'] = np.arange(len(current_dataset['inputs']))

        else:
            raise ValueError("input_structure must be either 'independence' or 'merge'")

    def __getitem__(self, idx, model_key=None, area_grp_id=None):
        """
        특정 인덱스의 데이터를 반환합니다.

        Args:
            idx: 데이터 인덱스.
            model_key: 'ECMWF' 또는 'LDAPS'.
            area_grp_id: 발전소 ID.

        Returns:
            dict: 입력 및 타깃 데이터.
        """
        if model_key is None or area_grp_id is None:
            raise ValueError("Both 'model_key' and 'area_grp_id' must be provided")

        model_key = str(model_key)
        area_grp_id = str(area_grp_id)

        dataset = self.datasets[model_key][area_grp_id]
        i = dataset['valid_idxs'][idx]

        # 입력 데이터 가져오기
        input_features = dataset['inputs'].iloc[i].values.astype(np.float32)

        # 타깃 데이터 가져오기
        target_value = dataset['targets'].iloc[i]
        return {
            'inputs': torch.tensor(input_features, dtype=torch.float32),
            'targets': torch.tensor([target_value], dtype=torch.float32)
        }

    def __len__(self, model_key=None, area_grp_id=None):
        """
        데이터 길이를 반환합니다.

        Args:
            model_key: 'ECMWF' 또는 'LDAPS'.
            area_grp_id: 발전소 ID.

        Returns:
            int: 데이터 길이.
        """
        if model_key is None or area_grp_id is None:
            total_length = 0
            for mk, area_groups in self.datasets.items():
                for ag_id in area_groups.keys():
                    total_length += len(self.datasets[mk][ag_id]['valid_idxs'])
            return total_length

        model_key = str(model_key)
        area_grp_id = str(area_grp_id)
        return len(self.datasets[model_key][area_grp_id]['valid_idxs'])


class FilteredDataset(Dataset):
    def __init__(self, dataset, model_key, area_grp_id):
        """
        특정 모델과 발전소 ID에 대한 필터링된 데이터셋.

        Args:
            dataset: WeatherPredictionDataset 객체.
            model_key: 'ECMWF' 또는 'LDAPS'.
            area_grp_id: 발전소 ID.
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
    DataLoader 생성.

    Args:
        dataset: WeatherPredictionDataset 객체.
        batch_size: 배치 크기.
        shuffle: 데이터 셔플 여부.
        num_workers: 데이터 로딩 워커 수.
        pin_memory: 고정 메모리 사용 여부.
        drop_last: 마지막 배치를 버릴지 여부.

    Returns:
        dict: 모델 및 발전소 ID별 DataLoader.
    """
    data_loaders = {}

    for model_key, area_groups in dataset.datasets.items():
        for area_grp_id in area_groups.keys():
            # 필터링된 데이터셋 생성
            filtered_dataset = FilteredDataset(dataset, model_key, area_grp_id)
            # DataLoader 생성
            data_loaders[(model_key, area_grp_id)] = DataLoader(filtered_dataset,
                                                                batch_size=batch_size,
                                                                shuffle=shuffle,
                                                                num_workers=num_workers,
                                                                pin_memory=pin_memory,
                                                                drop_last=drop_last)
    return data_loaders