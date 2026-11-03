# SNU-AI-Prediction

수치예보(NWP) 기상 데이터로 **신재생 발전소의 시간별 발전량**을 예측하는 PyTorch·PyCaret 기반 프레임워크.

- 범주: 태양광(`solar`), 풍력(`wind`)
- 같은 입력으로 딥러닝 시계열 모델과 머신러닝 회귀 모델(앙상블 포함) 비교
- 저장소에는 **데이터와 학습 결과가 포함되어 있지 않음** → [데이터 형식](#데이터-형식) 참고
- 기본 가정: "발전소별 여러 수치예보 모델의 예보값 + 같은 시각의 실측 발전량"

## 목차

1. [개요](#개요)
2. [설치](#설치)
3. [데이터 형식](#데이터-형식)
4. [전처리와 분할](#전처리와-분할)
5. [실행](#실행)
6. [지원 모델](#지원-모델)
7. [출력](#출력)
8. [디렉터리 구조](#디렉터리-구조)

## 개요

```text
수치예보 CSV (예보원 A, B …)  ─┐
                               ├─▶ 발전소·시각 기준 정렬 ─▶ 학습/검증/시험 분할 ─▶ 입력 표준화
실측 발전량 CSV ───────────────┘                                                      │
                                                                                     ▼
                                  딥러닝(DLinear·PatchTST·iTransformer·TimesNet …)  또는  PyCaret 회귀/앙상블
                                                                                     │
                                                                                     ▼
                                        발전소·지역·전체 단위 지표(CSV/JSON)와 예측값(.npy) 저장
```

- 입력: 예보 대상 시각(`FCST_TM`)의 기상 변수 (기온, 습도, 일사량, 풍속·풍향 등)
- 출력: 같은 시각의 발전량(`QGEN`)
- 학습 단위: 발전소(태양광) 또는 호기(풍력 터빈 1기)마다 별도 모델

## 설치

```bash
git clone https://github.com/CEDPProject/snu-ai-prediction.git
cd snu-ai-prediction
pip install -r requirements.txt
```

- 권장 환경: Python 3.8~3.10, CUDA GPU 1장 (CPU도 가능하나 딥러닝 모델은 느림)
- `wandb`: import만 필요. 기본 설정(`TRAIN.wandb.use: false`)에서는 로그인 불필요

## 데이터 형식

### 폴더 구조

```text
<datadir>/
└── <data_version>/                        # 예: version2_241107
    ├── 태양광발전단지/                     # category_name=solar
    │   ├── 태양광발전단지_실적데이터.csv          # 실측 발전량 (cp949)
    │   ├── 태양광발전단지_위치정보.csv            # 발전소 목록 (cp949)
    │   ├── 태양광발전단지_ECMWF_수치예보데이터.csv # 예보원 A (utf-8)
    │   ├── 태양광발전단지_LDAPS_수치예보데이터.csv # 예보원 B (utf-8)
    │   └── 태양광발전단지_기상관측데이터.csv       # 지상 관측 (읽기만 하고 미사용)
    └── 풍력발전단지/                       # category_name=wind
        ├── 풍력발전단지_실적데이터.csv            # (cp949)
        ├── 풍력발전단지_호기별_위치정보.csv        # 호기(터빈) 목록 (cp949)
        ├── 풍력발전단지_호기별_ECMWF_수치예보데이터.csv  # (cp949)
        ├── 풍력발전단지_호기별_LDAPS_수치예보데이터.csv  # (cp949)
        └── 풍력발전단지_기상관측데이터.csv
```

| 설정 | 기본값 | 설명 |
|---|---|---|
| `DATASET.datadir` | `../../data/ecobrain` | 데이터 루트 (`src/` 기준 상대경로) |
| `DATASET.data_version` | `version2_241107` | `<datadir>` 아래 하위 폴더 |
| `DATASET.category_name` | `wind` | `solar` 또는 `wind` |

- 데이터는 저장소 밖에 두고 실행 시 덮어쓰기 권장: `--opts "DATASET.datadir=/path/to/data"`
- 파일 이름·인코딩 정의 위치: `src/data_provider/load_dataset.py`의 `load_dataset()` 앞부분

### 수치예보 파일 (예보원별 1개)

한 행 = "어떤 발전소(호기)에 대해, 어느 시각에 발표된 예보가, 어느 시각을 예측한 값".

| 컬럼 | 의미 | 예시 | 비고 |
|---|---|---|---|
| `AREA_GRP_ID` | 발전소/호기 ID | `5375`, `4593_001` | 위치정보 파일의 ID와 일치해야 함 |
| `CRTN_TM` | 예보 발표 시각 | `2022-01-01 00:00:00` | `%Y-%m-%d %H:%M:%S` |
| `FCST_TM` | 예보 대상 시각 | `2022-01-01 06:00:00` | 실측 발전량의 `TM`과 매칭 |
| `LEAD_TM` | 선행시간(시간) | `6` | 기본 설정에서 입력 제외 |
| `NWP_TP_CD` | 예보원 코드 | `22`, `2` | 기본 설정에서 입력 제외 |
| 기상 변수 | 모델 입력 | `TEM, REH, PSFC, SLPR, SWDOWN, TCDC, RAIN, U10, V10, WS10, WD10, …` | 위 컬럼과 `ignore_feature`를 뺀 **나머지 수치 컬럼 전부가 입력** |

- 시간 간격: **1시간** 가정 (모델 설정 `freq: h`)
- 같은 `FCST_TM`에 여러 발표 예보가 있으면 `lead_time_align: True`(기본)일 때 **가장 최근 발표(최소 `LEAD_TM`)** 1행만 유지
- `align_time: True`: 두 예보원에 모두 있는 `(CRTN_TM, FCST_TM)` 쌍만 사용

### 실측 발전량 파일

| 컬럼 | 의미 | 비고 |
|---|---|---|
| `ID` | 발전소 ID | 풍력은 발전단지 ID(`4593`). 호기 ID의 `_` 앞부분과 매칭 |
| `NAME` | 발전소 이름 | 로드 후 삭제 |
| `TM` | 측정 시각 | `FCST_TM`과 같은 형식 |
| `QGEN` | 발전량 | **예측 대상**. 결측은 앞 값으로 채움 |

### 위치정보 파일

학습할 발전소/호기 목록. ID 열만 사용.

- 태양광: `GEN_ID, GEN_NM, LATITD, LONGTD, RTD_CAPA, …` → `GEN_ID`에 있는 ID만 사용
- 풍력: `UNIT_ID, UNIT_NM, LATITD, LONGTD, RTD_CAPA, GEN_ID, HUB_HEIGHT` → `UNIT_ID`에 있는 호기만 사용

### 내 데이터에 적용하기

1. **CSV를 위 스키마로 변환** (권장, 코드 수정 없음)
   - 예보원이 하나뿐이면 같은 파일을 두 예보원 이름으로 복사하거나 `load_dataset.py`에서 하나 제거
   - 실측 데이터에 발전소 이름이 없으면 `NAME` 열을 빈 값으로 둠
   - 위치정보는 ID 열만 있어도 됨
2. **로더 수정**이 필요한 경우 수정 위치
   - 파일 이름·인코딩: `load_dataset.py`의 `load_dataset()` 앞부분
   - 예보원 이름(`ECMWF`, `LDAPS`): `load_dataset.py`, `build_dataset.py`
   - 키·타깃 열 이름(`AREA_GRP_ID`, `FCST_TM`, `TM`, `QGEN`): `load_dataset.py`, `build_dataset.py`, `exp_builder_ml.py`
   - 풍력 `발전단지_호기` ID 규칙(`split('_')`): `load_dataset.py`, `exp_builder_dl.py`, `exp_builder_ml.py`
   - 범주 이름(`solar`, `wind`): `src/data_provider/factory.py`의 `assert`
   - 지역별 설비용량(평가 시 용량 10% 미만 구간 제외에 사용): `src/utils/utils.py`의 `capacity_dict`
3. **분할 기간 변경**: `split_date`/`end_date`가 2022~2023년 데이터 기준이므로 데이터 기간에 맞게 변경

## 전처리와 분할

| 설정 (`DATASET.*`) | 기본값 | 설명 |
|---|---|---|
| `target` | `power` | 예측 대상. 실측 발전량(`QGEN`)만 지원, 다른 값은 오류 |
| `input_structure` | `independence` | `independence`: 예보원별 별도 학습. `merge`: 같은 발전소·`FCST_TM`의 두 예보원 변수를 열 방향으로 합쳐(`*_ecmwf`, `*_ldaps`) 모델 하나로 학습 (결과 키 `merged`) |
| `individual_model` | `True` | `(예보원, 발전소)`마다 별도 모델 |
| `scale` | `standard` | 입력만 학습 구간 기준 스케일링 (`standard`, `minmax`, `minmax_square`), 타깃은 원 단위 유지 |
| `split_date` | `2023-01-01 00:00:00` | 이전은 학습(+검증), 이후 `end_date`까지는 시험 |
| `split_rate` | `[0.8, 0.0, 0.1]` | 학습·검증·시험 비율. 검증 0이면 검증 없음. `split_date`가 없으면 이 비율로 시간 순 분할 |
| `seq_len` | `12` | **입력 window 길이(시간)**. 예측 시점 수가 아님 |
| `target_type` | `multi` | `multi`: window 각 시각의 발전량 동시 예측 (`pred_len`을 `seq_len`과 같게 설정). `uni`: window 다음 한 시각만 예측 |
| `ignore_feature` | `['LEAD_TM', 'NWP_TP_CD']` | 입력에서 뺄 컬럼 |

- `TRAIN.clamping: True`: 예측값을 0 이상으로 자름 (발전량은 음수 없음)
- 머신러닝 모델(`ML*`): window 없이 시각별 1행 사용 → `seq_len` 영향 없음

## 실행

- 모든 명령은 **`src/`에서** 실행 (설정·데이터 경로가 `src/` 기준 상대경로)
- 설정 덮어쓰기: `--opts "키=값"` 여러 개 지정 가능, 값은 YAML로 해석 (예: `"TRAIN.epoch=50"`)

```bash
cd src

# 딥러닝 모델 1개
python main.py \
    --model_name DLinear \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    --opts "DATASET.category_name=solar" "DATASET.seq_len=24" "DATASET.pred_len=24"

# PyCaret 회귀·앙상블
python main.py \
    --model_name MLpycaret \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    --opts "DATASET.category_name=solar" "DATASET.seq_len=1" "MODELSETTING.ensemble.enable=True"
```

| 스크립트 | 내용 |
|---|---|
| `bash run_test.sh` | DLinear·PatchTST·iTransformer × solar/wind × seq_len 1/12/24, 이어서 PyCaret 앙상블 |
| `bash scripts/solar/all.sh`, `bash scripts/wind/all.sh` | 범주별 딥러닝 모델 전체 |
| `bash scripts/MLpycaret_all.sh` | PyCaret 모델 전체 |
| `scripts/Archive/` | 이전 실험용 스크립트 |

- 스크립트마다 `export CUDA_VISIBLE_DEVICES=...` 지정 → 사용할 GPU 번호에 맞게 수정

## 지원 모델

| 구분 | `--model_name` | 비고 |
|---|---|---|
| 딥러닝 | `DLinear`, `PatchTST`, `iTransformer`, `TimesNet` | 시계열 모델 |
| 딥러닝 | `NCART`, `TabM` | 표 형식 모델 |
| 머신러닝 | `MLModelEnsemble` | scikit-learn SVR·DT·RF·Ridge (`selected_models`) |
| 머신러닝 | `MLpycaret` | PyCaret 회귀(기본 `rf, et, gbr, lightgbm`), `blend`·`stack`·`bagging` 앙상블, 선택적 하이퍼파라미터 탐색(`enable_tuning`, `model_grids`) |

- 모델 하이퍼파라미터: `src/configs/model_setting.yaml`의 모델 이름별 항목
- 새 모델 추가: `src/models/`에 구현 → `src/models/__init__.py`에 등록 → `model_setting.yaml`에 같은 이름으로 설정 추가

## 출력

`src/saved_model/<data_version>/<model_name>/<exp_name>/version<N>/` (git 제외)

| 파일 | 내용 |
|---|---|
| `configs.yaml` | 실행에 쓰인 전체 설정 |
| `test_results.json` | 시험 구간 요약 지표 |
| `*_all_results.csv`, `*_area_level_results.csv`, `*_region_level_results.csv` | 발전소·지역·전체 단위 지표 (딥러닝) |
| `train_results.csv`, `test_*.csv` | 모델별 지표 (PyCaret) |
| `*_total_outputs.npy`, `*_total_targets.npy` | 예측값, 실측값 |
| `best_model_<예보원>_<ID>.pt` | 학습된 가중치 (`DEFAULT.del_pt: True`이면 실행 후 삭제) |

- 예측 그래프: `src/jupyter/plot_predictions.ipynb`

## 디렉터리 구조

```text
snu-ai-prediction/
├── requirements.txt
└── src/
    ├── main.py                 # 진입점: 설정 로드 → 데이터 → 학습/평가
    ├── arguments.py            # --default_cfg, --model_cfg, --opts 파싱
    ├── exp_builder_dl.py       # 딥러닝 학습·평가 루프
    ├── exp_builder_ml.py       # scikit-learn / PyCaret 학습·평가
    ├── run_test.sh             # 일괄 실험
    ├── configs/                # default_setting.yaml(데이터·학습), model_setting.yaml(모델)
    ├── data_provider/          # load_dataset.py(로드·정렬·분할·스케일링), build_dataset.py(window), factory.py
    ├── models/                 # 모델 구현과 등록(__init__.py)
    ├── layers/                 # 모델 공용 레이어
    ├── losses/, optimizers/    # 손실·옵티마이저 factory
    ├── utils/                  # 로그, 저장, 지표
    ├── scripts/                # solar/, wind/, Archive/ 실행 스크립트
    └── jupyter/                # 결과 시각화 노트북
```

## 참고 구현

- `DLinear`, `PatchTST`, `iTransformer`, `TimesNet` 등과 `src/layers/`:
  [Time-Series-Library](https://github.com/thuml/Time-Series-Library) (MIT) 기반
- `TabM`: [yandex-research/tabm](https://github.com/yandex-research/tabm) (Apache-2.0) 기반

## 라이선스

[MIT License](LICENSE)

