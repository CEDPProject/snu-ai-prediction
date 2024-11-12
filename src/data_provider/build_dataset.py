import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class WeatherPredictionDataset(Dataset):
    def __init__(self, data_dict: dict, ground_truth: pd.DataFrame,
                 category_name: str, input_structure: str = 'independence',
                 type: str = 'train'):

        self.datasets = {}  # 각 호기별 데이터셋을 저장할 딕셔너리

        # 데이터 구성
        if input_structure == 'independence':
            for model_key in ['ECMWF', 'LDAPS']:
                self.datasets[model_key] = []

                for entry_idx, entry in enumerate(data_dict[model_key]):
                    df = entry[type].copy()

                    # wind의 경우 AREA_GRP_ID에서 '4593_001'과 같은 값 중 '4593'만 사용
                    if category_name == 'wind':
                        df['AREA_GRP_ID'] = df['AREA_GRP_ID'].astype(str).str.split('_').str[0]

                    # ground_truth와 병합하여 타겟 추출
                    merged = pd.merge(
                        df,
                        ground_truth,
                        left_on=['AREA_GRP_ID', 'FCST_TM'],
                        right_on=['ID', 'TM'],
                        how='inner'
                    )

                    # inputs와 targets 구성
                    inputs = merged.drop(columns=['QGEN', 'ID', 'TM'])
                    targets = merged['QGEN']

                    # 각 호기별 데이터셋 저장
                    self.datasets[model_key].append({
                        'inputs': inputs.reset_index(drop=True),
                        'targets': targets.reset_index(drop=True)
                    })

        elif input_structure == 'merge':
            self.datasets['merge'] = []

            for entry_idx, entry in enumerate(data_dict['merge']):
                df = entry[type].copy()

                # wind의 경우 AREA_GRP_ID에서 '4593_001'과 같은 값 중 '4593'만 사용
                if category_name == 'wind':
                    df['AREA_GRP_ID'] = df['AREA_GRP_ID'].astype(str).str.split('_').str[0]

                # ground_truth와 병합하여 타겟 추출
                merged = pd.merge(
                    df,
                    ground_truth,
                    left_on=['AREA_GRP_ID', 'FCST_TM'],
                    right_on=['ID', 'TM'],
                    how='inner'
                )

                # inputs와 targets 구성
                inputs = merged.drop(columns=['QGEN', 'ID', 'TM'])
                targets = merged['QGEN']

                # 병합된 데이터 저장
                self.datasets['merge'].append({
                    'inputs': inputs.reset_index(drop=True),
                    'targets': targets.reset_index(drop=True)
                })

        else:
            raise ValueError("input_structure must be either 'independence' or 'merge'")

        # 각 데이터셋의 유효 인덱스 생성
        for key in self.datasets:
            for dataset in self.datasets[key]:
                dataset['valid_idxs'] = np.arange(len(dataset['inputs']))

    def __getitem__(self, idx, model_key='ECMWF', entry_idx=0):
        dataset = self.datasets[model_key][entry_idx]
        i = dataset['valid_idxs'][idx]

        # Input features
        input_features = np.array(dataset['inputs'].iloc[i])

        # Target value
        target_value = np.array(dataset['targets'].iloc[i])

        item = {
            'inputs': torch.Tensor(input_features),
            'targets': torch.Tensor([target_value])  # Ensure target is a tensor with a single value
        }

        return item

    def __len__(self, model_key='ECMWF', entry_idx=0):
        return len(self.datasets[model_key][entry_idx]['valid_idxs'])