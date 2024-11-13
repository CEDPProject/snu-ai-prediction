import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pdb
import warnings
from sklearn.model_selection import train_test_split
from collections import Counter

warnings.filterwarnings('ignore')

def load_dataset(
                datadir: str,
                category_name: str,                
                scale: str,
                input_structure: str,
                data_version: str = 'version2_241107',
                align_time: bool = True,
                pre_training: bool = False,
                lead_time_align: bool = False,
                del_feature: list = ['SWDDIR'],
                split_rate: list = [0.8, 0.1, 0.1]):
    
    # 태양광 발전량
    if category_name == 'solar':
        base_path = os.path.join("../../data/ecobrain/", data_version, "태양광발전단지")
        data_dict = {
            'weather_forecast': pd.read_csv(os.path.join(base_path, "태양광발전단지_기상관측데이터.csv")),
            'real_power': pd.read_csv(os.path.join(base_path, "태양광발전단지_실적데이터.csv"), encoding='cp949'),
            'location': pd.read_csv(os.path.join(base_path, "태양광발전단지_위치정보.csv"), encoding='cp949'),
            'ECMWF': pd.read_csv(os.path.join(base_path, "태양광발전단지_ECMWF_수치예보데이터.csv")),
            'LDAPS': pd.read_csv(os.path.join(base_path, "태양광발전단지_LDAPS_수치예보데이터.csv")),
        }

    if category_name == 'wind':
        base_path = os.path.join("../../data/ecobrain/", data_version, "풍력발전단지")
        data_dict = {
            'weather_forecast': pd.read_csv(os.path.join(base_path, "풍력발전단지_기상관측데이터.csv")),
            'real_power': pd.read_csv(os.path.join(base_path, "풍력발전단지_실적데이터.csv"), encoding='cp949'),
            'location': pd.read_csv(os.path.join(base_path, "풍력발전단지_호기별_위치정보.csv"), encoding='cp949'),
            'ECMWF': pd.read_csv(os.path.join(base_path, "풍력발전단지_호기별_ECMWF_수치예보데이터.csv"), encoding='cp949'),
            'LDAPS': pd.read_csv(os.path.join(base_path, "풍력발전단지_호기별_LDAPS_수치예보데이터.csv"), encoding='cp949'),
            # 'ECMWF': pd.read_csv(os.path.join(base_path, "풍력발전단지_ECMWF_수치예보데이터.csv")),
            # 'LDAPS': pd.read_csv(os.path.join(base_path, "풍력발전단지_LDAPS_수치예보데이터.csv")),
        }

    ground_truth = data_dict['real_power'].drop(columns=['NAME'])
    
    # 발전 호기 필터링
    data_dict['ECMWF'] = filtering_id(data_dict['ECMWF'], data_dict['location'], category_name)
    data_dict['LDAPS'] = filtering_id(data_dict['LDAPS'], data_dict['location'], category_name)
    
    # 사용하지 않는 Feature 제거
    data_dict['ECMWF'] = remove_columns(data_dict['ECMWF'], del_feature)
    data_dict['LDAPS'] = remove_columns(data_dict['LDAPS'], del_feature)

    # 동일 CRTN_TM, FCST_TM 가지는 데이터만 모델링 진행
    if align_time:
        data_list, common_times = align_times([data_dict['ECMWF'], data_dict['LDAPS']])
        data_dict['ECMWF'], data_dict['LDAPS'] = data_list
    
    # 위치(호기)별로 데이터 분할
    data_dict['ECMWF'] = [group.reset_index(drop=True) for _, group in data_dict['ECMWF'].groupby('AREA_GRP_ID')]
    data_dict['LDAPS'] = [group.reset_index(drop=True) for _, group in data_dict['LDAPS'].groupby('AREA_GRP_ID')]

    # FCST_TM 이 동일한 경우, LEAD_TM 기준으로 전처리
    if lead_time_align:
        data_dict['ECMWF'], data_dict['LDAPS'] = process_and_sort_dfs([data_dict['ECMWF'], data_dict['LDAPS']])
    else:
        data_dict['ECMWF'] = [df.sort_values(by=["CRTN_TM", "FCST_TM"]).reset_index(drop=True) for df in data_dict['ECMWF']]
        data_dict['LDAPS'] = [df.sort_values(by=["CRTN_TM", "FCST_TM"]).reset_index(drop=True) for df in data_dict['LDAPS']]
    
    # 데이터 Input 구조 설정 및 Train/val/test 분할, 전처리
    data_dict, scaler_dict = split_processing(data_dict, 
                                    input_structure = input_structure, 
                                    split_rate      = split_rate, 
                                    scale           = scale, 
                                    pre_training    = pre_training)
    
    features = data_dict['ECMWF']['5375']['train'].columns
    print(f"ECMWF 발전단지 호기 수: {len(data_dict['ECMWF'])} | LDAPS 발전단지 호기 수: {len(data_dict['LDAPS'])}")
    print(f"Train length: {len(data_dict['ECMWF']['5375']['train'])} | Val length: {len(data_dict['ECMWF']['5375']['val'])} | Test length: {len(data_dict['ECMWF']['5375']['test'])}")
    
    return data_dict, ground_truth, features, scaler_dict


# 발전 호기 필터링
def filtering_id(fcst, location, category_name):
    if category_name == 'solar':
        unit_ids = location['GEN_ID'].unique()
    elif category_name == 'wind':
        unit_ids = location['UNIT_ID'].unique()
    filtered_wind_ecmwf = fcst[fcst['AREA_GRP_ID'].isin(unit_ids)]
    return filtered_wind_ecmwf

# 동일 CRTN_TM, FCST_TM 가지는 데이터만 모델링 진행
def align_times(data_list):
    # 데이터 포맷 변환
    for i, df in enumerate(data_list):
        df['CRTN_TM'] = pd.to_datetime(df['CRTN_TM'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
        df['FCST_TM'] = pd.to_datetime(df['FCST_TM'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
        data_list[i] = df.dropna(subset=['CRTN_TM', 'FCST_TM']).reset_index(drop=True)

    time_sets = [
        set(df[['CRTN_TM', 'FCST_TM']].drop_duplicates().apply(tuple, axis=1)) for df in data_list
    ]

    common_times = sorted(set.intersection(*time_sets))

    filtered_data_list = [
        df[df[['CRTN_TM', 'FCST_TM']].apply(tuple, axis=1).isin(common_times)].reset_index(drop=True)
        for df in data_list
    ]
    
    return filtered_data_list, common_times

# 사용하지 않는 Feature 제거
def remove_columns(df, columns):
    df = df.drop(columns=[col for col in columns if col in df.columns], errors='ignore')
    return df

# LEAD_TM 기준으로 데이터 전처리
def process_and_sort_dfs(dfs):
    processed_dfs = []
    
    for df in dfs:
        df = df.loc[df.groupby('FCST_TM')['LEAD_TM'].idxmin()]
        df = df.sort_values(by='CRTN_TM').reset_index(drop=True)
        
        processed_dfs.append(df)
    return processed_dfs


def split_processing(data_dict, test_size=0.2, random_state=42):
    result_dict = {}

    for key, df_list in data_dict.items():
        result_dict[key] = []
        
        for df in df_list:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
            scaler = MinMaxScaler(feature_range=(0, 1))
            
            train_df_normalized = train_df.copy()
            test_df_normalized = test_df.copy()
            
            cols_to_normalize = df.columns.difference(['AREA_GRP_ID', 'CRTN_TM', 'FCST_TM'])
            
            train_df_normalized[cols_to_normalize] = scaler.fit_transform(train_df[cols_to_normalize])
            test_df_normalized[cols_to_normalize] = scaler.transform(test_df[cols_to_normalize])
            
            result_dict[key].append({
                'train': train_df_normalized,
                'test': test_df_normalized,
                'scaler': scaler
            })
    return result_dict

# 데이터 Input 구조 설정 및 Train/val/test 분할, 전처리
def split_processing(data_dict, 
                     input_structure='independence', 
                     split_rate=[0.8, 0.1, 0.1], 
                     scale='minmax', 
                     pre_training=False, 
                     random_state=42):
    
    scaler_map = {
        'minmax': MinMaxScaler(feature_range=(-1, 1)),
        'minmax_square': MinMaxScaler(),
        'standard': StandardScaler()
    }
    scaler_type = scaler_map[scale]
    result_dict = {}  # 결과를 저장할 딕셔너리
    scaler_dict = {}  # AREA_GRP_ID 별로 스케일러를 저장할 딕셔너리

    relevant_keys = ['ECMWF', 'LDAPS']
    filtered_data_dict = {key: data_dict[key] for key in relevant_keys if key in data_dict}

    if input_structure == 'independence':
        for key, df_list in filtered_data_dict.items():
            result_dict[key] = {}
            
            for df in df_list:
                area_grp_ids = df['AREA_GRP_ID'].unique()
                
                for area_grp_id in area_grp_ids:
                    area_df = df[df['AREA_GRP_ID'] == area_grp_id].copy()
                    
                    scaler = scaler_type  # 각 그룹마다 새로운 스케일러 생성
                    cols_to_normalize = area_df.columns.difference(['AREA_GRP_ID', 'CRTN_TM', 'FCST_TM'])
                    
                    if pre_training:
                        # Pre-training: Train 데이터만 사용
                        train_df = area_df.copy()
                        train_df[cols_to_normalize] = scaler.fit_transform(train_df[cols_to_normalize]) ** (2 if scale == 'minmax_square' else 1)
                        result_dict[key][str(area_grp_id)] = {'train': train_df, 'scaler': scaler}
                    else:
                        # Train/Validation/Test Split
                        train_size = split_rate[0]
                        val_size = split_rate[1] / (1 - train_size)  # Validation 비율 조정
                        
                        train_df, temp_df = train_test_split(area_df, test_size=1-train_size, random_state=random_state, shuffle=False)
                        val_df, test_df = train_test_split(temp_df, test_size=1-val_size, random_state=random_state, shuffle=False)
                        
                        train_df[cols_to_normalize] = scaler.fit_transform(train_df[cols_to_normalize]) ** (2 if scale == 'minmax_square' else 1)
                        val_df[cols_to_normalize] = scaler.transform(val_df[cols_to_normalize]) ** (2 if scale == 'minmax_square' else 1)
                        test_df[cols_to_normalize] = scaler.transform(test_df[cols_to_normalize]) ** (2 if scale == 'minmax_square' else 1)
                        
                        result_dict[key][str(area_grp_id)] = {
                            'train': train_df,
                            'val': val_df,
                            'test': test_df,
                            'scaler': scaler
                        }
                    
                    # 스케일러 저장
                    scaler_dict[(key, str(area_grp_id))] = scaler

    elif input_structure == 'merge':
        result_dict['merged'] = {}
        merged_list = zip(data_dict['ECMWF'], data_dict['LDAPS'])
        
        for ecmwf_df, ldaps_df in merged_list:
            merged_df = pd.merge(ecmwf_df, ldaps_df, on=['AREA_GRP_ID', 'CRTN_TM', 'FCST_TM'], suffixes=('_ecmwf', '_ldaps'))
            
            area_grp_ids = merged_df['AREA_GRP_ID'].unique()
            
            for area_grp_id in area_grp_ids:
                area_df = merged_df[merged_df['AREA_GRP_ID'] == area_grp_id].copy()
                
                scaler = scaler_type  # 각 그룹마다 새로운 스케일러 생성
                cols_to_normalize = area_df.columns.difference(['AREA_GRP_ID', 'CRTN_TM', 'FCST_TM'])
                
                if pre_training:
                    train_df = area_df.copy()
                    train_df[cols_to_normalize] = scaler.fit_transform(train_df[cols_to_normalize]) ** (2 if scale == 'minmax_square' else 1)
                    result_dict['merged'][str(area_grp_id)] = {'train': train_df, 'scaler': scaler}
                else:
                    train_size = split_rate[0]
                    val_size = split_rate[1] / (1 - train_size)
                    
                    train_df, temp_df = train_test_split(area_df, test_size=1-train_size, random_state=random_state, shuffle=False)
                    val_df, test_df = train_test_split(temp_df, test_size=1-val_size, random_state=random_state, shuffle=False)
                    
                    train_df[cols_to_normalize] = scaler.fit_transform(train_df[cols_to_normalize]) ** (2 if scale == 'minmax_square' else 1)
                    val_df[cols_to_normalize] = scaler.transform(val_df[cols_to_normalize]) ** (2 if scale == 'minmax_square' else 1)
                    test_df[cols_to_normalize] = scaler.transform(test_df[cols_to_normalize]) ** (2 if scale == 'minmax_square' else 1)
                    
                    result_dict['merged'][str(area_grp_id)] = {
                        'train': train_df,
                        'val': val_df,
                        'test': test_df,
                        'scaler': scaler
                    }
                
                # 스케일러 저장
                scaler_dict[('merged', str(area_grp_id))] = scaler
    else:
        raise ValueError("Mode should be either 'independence' or 'merge'")
    
    return result_dict, scaler_dict