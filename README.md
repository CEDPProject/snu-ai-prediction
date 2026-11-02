# SNU-AI-Prediction

수치예보(NWP) 기상 데이터로 **신재생 발전소의 시간별 발전량**을 예측하는 PyTorch·PyCaret 기반 프레임워크입니다.
태양광(`solar`)과 풍력(`wind`) 두 범주를 지원하며, 같은 입력으로 딥러닝 시계열 모델과 머신러닝 회귀 모델(앙상블 포함)을
비교할 수 있습니다.

이 저장소에는 **데이터와 학습 결과가 포함되어 있지 않습니다.** 아래 [데이터 형식](#-데이터-형식)에 맞춰 직접 준비한
데이터를 사용하세요. 기본 설정은 "발전소별 여러 수치예보 모델의 예보값 + 같은 시각의 실측 발전량" 형태의 데이터를
가정합니다.

## 📌 개요

```
수치예보 CSV (예보원 A, B …)  ─┐
                               ├─▶ 발전소·시각 기준 정렬 ─▶ 학습/검증/시험 분할 ─▶ 입력 표준화
실측 발전량 CSV ───────────────┘                                                      │
                                                                                     ▼
                                  딥러닝(DLinear·PatchTST·iTransformer·TimesNet …)  또는  PyCaret 회귀/앙상블
                                                                                     │
                                                                                     ▼
                                        발전소·지역·전체 단위 지표(CSV/JSON)와 예측값(.npy) 저장
```

- **입력**: 예보 시각(`FCST_TM`)의 기상 변수(기온, 습도, 일사량, 풍속·풍향 등)
- **출력**: 같은 시각의 발전량(`QGEN`)
- **단위**: 발전소(태양광) 또는 호기(풍력, 터빈 1기)마다 별도 모델을 학습하는 것이 기본입니다.

## 🚀 설치

```bash
git clone https://github.com/CEDPProject/snu-ai-prediction.git
cd snu-ai-prediction
pip install -r requirements.txt
```

- Python 3.8~3.10, CUDA GPU 1장을 권장합니다. CPU에서도 동작하지만 딥러닝 모델은 느립니다.
- `wandb`는 import만 하며, 기본 설정(`TRAIN.wandb.use: false`)에서는 로그인하지 않아도 됩니다.

## 📂 데이터 형식

### 1. 폴더 구조

데이터 위치는 `src/configs/default_setting.yaml`의 `DATASET.datadir`(기본값 `../../data/ecobrain`, `src/` 기준
상대경로)와 `DATASET.data_version`으로 정합니다. 데이터는 저장소 밖에 두는 것을 권장합니다.

```
<datadir>/
└── <data_version>/                        # 예: version2_241107
    ├── 태양광발전단지/                     # category_name=solar
    │   ├── 태양광발전단지_실적데이터.csv          # 실측 발전량 (cp949)
    │   ├── 태양광발전단지_위치정보.csv            # 발전소 목록 (cp949)
    │   ├── 태양광발전단지_ECMWF_수치예보데이터.csv # 예보원 A (utf-8)
    │   ├── 태양광발전단지_LDAPS_수치예보데이터.csv # 예보원 B (utf-8)
    │   └── 태양광발전단지_기상관측데이터.csv       # 지상 관측 (읽기만 하고 현재 미사용)
    └── 풍력발전단지/                       # category_name=wind
        ├── 풍력발전단지_실적데이터.csv            # (cp949)
        ├── 풍력발전단지_호기별_위치정보.csv        # 호기(터빈) 목록 (cp949)
        ├── 풍력발전단지_호기별_ECMWF_수치예보데이터.csv  # (cp949)
        ├── 풍력발전단지_호기별_LDAPS_수치예보데이터.csv  # (cp949)
        └── 풍력발전단지_기상관측데이터.csv
```

다른 위치를 쓰려면 실행할 때 `--opts "DATASET.datadir=/path/to/data"`처럼 덮어쓰면 됩니다.
파일 이름과 인코딩은 `src/data_provider/load_dataset.py`의 `load_dataset()` 앞부분에 정의되어 있습니다.

### 2. 수치예보 파일 (예보원별 1개)

한 행은 "어떤 발전소(또는 호기)에 대해, 어느 시각에 발표된 예보가, 어느 시각을 예측한 값"입니다.

| 컬럼 | 의미 | 예시 | 비고 |
|---|---|---|---|
| `AREA_GRP_ID` | 발전소/호기 ID | `5375`, `4593_001` | 위치정보 파일의 ID와 같아야 함 |
| `CRTN_TM` | 예보 발표 시각 | `2022-01-01 00:00:00` | `%Y-%m-%d %H:%M:%S` |
| `FCST_TM` | 예보 대상 시각 | `2022-01-01 06:00:00` | 실측 발전량의 `TM`과 매칭 |
| `LEAD_TM` | 선행시간(시간) | `6` | 기본 설정에서 입력에서 제외 |
| `NWP_TP_CD` | 예보원 코드 | `22`, `2` | 기본 설정에서 입력에서 제외 |
| 기상 변수들 | 모델 입력 | `TEM, REH, PSFC, SLPR, SWDOWN, TCDC, RAIN, U10, V10, WS10, WD10, …` | 위 컬럼과 `ignore_feature`를 뺀 **나머지 수치 컬럼이 모두 입력** |

- 시간 간격은 **1시간**을 가정합니다(모델 설정의 `freq: h`).
- 같은 `FCST_TM`에 여러 발표 시각의 예보가 있으면, `lead_time_align: True`(기본)일 때 **가장 최근 발표(최소 `LEAD_TM`)** 한 행만 남깁니다.
- `align_time: True`로 두면 두 예보원에 모두 있는 `(CRTN_TM, FCST_TM)` 쌍만 사용합니다.

### 3. 실측 발전량 파일

| 컬럼 | 의미 | 비고 |
|---|---|---|
| `ID` | 발전소 ID | 풍력은 발전단지 ID(`4593`). 호기 ID의 `_` 앞부분과 매칭 |
| `NAME` | 발전소 이름 | 로드 후 삭제 |
| `TM` | 측정 시각 | `FCST_TM`과 같은 형식 |
| `QGEN` | 발전량 | **예측 대상**. 결측은 앞 값으로 채움(forward fill) |

### 4. 위치정보 파일

예보 파일에서 학습할 발전소/호기를 고르는 데 쓰입니다.

- 태양광: `GEN_ID, GEN_NM, LATITD, LONGTD, RTD_CAPA, …` — `GEN_ID`에 있는 ID만 사용
- 풍력: `UNIT_ID, UNIT_NM, LATITD, LONGTD, RTD_CAPA, GEN_ID, HUB_HEIGHT` — `UNIT_ID`에 있는 호기만 사용

### 5. 내 데이터에 맞추기

비슷한 형태(발전소별 기상 예보 + 시간별 발전량)라면 다음 중 하나로 적용할 수 있습니다.

1. **CSV를 위 스키마로 변환** (권장, 코드 수정 없음)
   - 예보원이 하나뿐이면, 같은 파일을 두 예보원 이름으로 복사하거나 `load_dataset.py`에서 하나를 빼세요.
   - 실측 데이터에 발전소 이름이 없으면 `NAME` 컬럼을 빈 값으로 두면 됩니다.
   - 위치정보는 ID 컬럼만 있어도 됩니다(나머지 컬럼은 사용하지 않음).
2. **로더 수정**
   - 파일 이름·인코딩: `src/data_provider/load_dataset.py` `load_dataset()` 앞부분
   - 예보원 이름(`ECMWF`, `LDAPS`): `load_dataset.py`, `build_dataset.py`
   - 키·타깃 컬럼 이름(`AREA_GRP_ID`, `FCST_TM`, `TM`, `QGEN`): `load_dataset.py`, `build_dataset.py`, `exp_builder_ml.py`
   - 풍력의 `발전단지_호기` ID 규칙(`split('_')`): `load_dataset.py`, `exp_builder_dl.py`, `exp_builder_ml.py`
   - 범주 이름(`solar`, `wind`): `src/data_provider/factory.py`의 `assert`
3. **분할 기간 변경**: `split_date`/`end_date`는 2022~2023년 데이터에 맞춰져 있으므로 데이터 기간에 맞게 바꾸세요.

## ⚙️ 전처리와 분할

| 설정 (`DATASET.*`) | 기본값 | 설명 |
|---|---|---|
| `category_name` | `wind` | `solar` 또는 `wind` |
| `input_structure` | `independence` | `independence`: 예보원별로 따로 학습. `merge`: 같은 발전소·예보 대상 시각(`FCST_TM`)의 두 예보원 변수를 열 방향으로 합쳐(`*_ecmwf`, `*_ldaps`) 모델 하나로 학습. 결과 키는 `merged` |
| `individual_model` | `True` | `(예보원, 발전소)`마다 별도 모델 |
| `target` | `power` | 예측 대상. 현재 실측 발전량(`QGEN`)만 지원하며 다른 값은 오류 |
| `scale` | `standard` | 입력만 학습 구간 기준으로 스케일링(`standard`, `minmax`, `minmax_square`). 타깃은 원 단위 유지 |
| `split_date` | `2023-01-01 00:00:00` | 이 시각 이전은 학습(+검증), 이후 `end_date`까지는 시험 |
| `split_rate` | `[0.8, 0.0, 0.1]` | 학습·검증·시험 비율. 검증 비율이 0이면 검증 없음. `split_date`가 없으면 이 비율로 시간 순 분할 |
| `seq_len` | `12` | **입력 윈도 길이(시간)**. 예측 시점 수가 아님 |
| `target_type` | `multi` | `multi`: 윈도 각 시각의 발전량을 동시에 예측(이때 `pred_len`을 `seq_len`과 같게 설정). `uni`: 윈도 다음 한 시각만 예측 |
| `ignore_feature` | `['LEAD_TM', 'NWP_TP_CD']` | 입력에서 뺄 컬럼 |

- 예측값은 `TRAIN.clamping: True`일 때 0 이상으로 자릅니다(발전량은 음수가 없음).
- 머신러닝 모델(`ML*`)은 윈도를 쓰지 않고 시각별 한 행을 그대로 사용하므로 `seq_len`의 영향을 받지 않습니다.

## 💻 실행

모든 명령은 **`src/`에서** 실행합니다. 설정 파일과 데이터 경로가 `src/` 기준 상대경로입니다.

```bash
cd src
```

### 딥러닝 모델 1개

```bash
python main.py \
    --model_name DLinear \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    --opts "DATASET.category_name=solar" "DATASET.seq_len=24"
```

`--opts`에는 `"키=값"`을 여러 개 줄 수 있고, 값은 YAML로 해석됩니다. 예: `"TRAIN.epoch=50"`, `"DATASET.datadir=/data/my_plants"`.

### PyCaret 회귀·앙상블

```bash
python main.py \
    --model_name MLpycaret \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    --opts "DATASET.category_name=solar" "DATASET.seq_len=1" \
           "MODELSETTING.ensemble.enable=True"
```

### 일괄 실행

```bash
bash run_test.sh                 # DLinear·PatchTST·iTransformer × solar/wind × seq_len 1/12/24, 이어서 PyCaret 앙상블
bash scripts/solar/all.sh        # 태양광 딥러닝 모델 전체
bash scripts/wind/all.sh         # 풍력 딥러닝 모델 전체
bash scripts/MLpycaret_all.sh    # PyCaret 모델 전체
```

스크립트마다 `export CUDA_VISIBLE_DEVICES=...`가 있으니 사용할 GPU 번호에 맞게 고치세요.
`scripts/Archive/`는 이전 실험용 스크립트입니다.

## 🧪 지원 모델

| 구분 | `--model_name` | 비고 |
|---|---|---|
| 딥러닝 | `DLinear`, `PatchTST`, `iTransformer`, `TimesNet` | 시계열 모델 |
| 딥러닝 | `NCART`, `TabM` | 표 형식 모델 |
| 머신러닝 | `MLModelEnsemble` | scikit-learn SVR·DT·RF·Ridge (`selected_models`) |
| 머신러닝 | `MLpycaret` | PyCaret 회귀 모델(`selected_models`, 기본 `rf, et, gbr, lightgbm`)과 `blend`·`stack`·`bagging` 앙상블, 선택적 하이퍼파라미터 탐색(`enable_tuning`, `model_grids`) |

모델 하이퍼파라미터는 `src/configs/model_setting.yaml`에 모델 이름별로 있습니다.

## 📤 출력

결과는 `src/saved_model/<data_version>/<model_name>/<exp_name>/version<N>/`에 저장되며 git에서 제외됩니다.

| 파일 | 내용 |
|---|---|
| `configs.yaml` | 실행에 쓰인 전체 설정 |
| `test_results.json` | 시험 구간 요약 지표 |
| `*_all_results.csv`, `*_area_level_results.csv`, `*_region_level_results.csv` | 발전소·지역·전체 단위 지표 (딥러닝) |
| `train_results.csv`, `test_*.csv` | 모델별 지표 (PyCaret) |
| `*_total_outputs.npy`, `*_total_targets.npy` | 예측값과 실측값 |
| `best_model_<예보원>_<ID>.pt` | 학습된 가중치 (`DEFAULT.del_pt: True`이면 실행 후 삭제) |

예측 그래프는 `src/jupyter/plot_predictions.ipynb`로 그릴 수 있습니다.

## 📁 프로젝트 구조

```
snu-ai-prediction/
├── requirements.txt
└── src/
    ├── main.py                 # 진입점: 설정 로드 → 데이터 → 학습/평가
    ├── arguments.py            # --default_cfg, --model_cfg, --opts 파싱
    ├── exp_builder_dl.py       # 딥러닝 학습·평가 루프
    ├── exp_builder_ml.py       # scikit-learn / PyCaret 학습·평가
    ├── run_test.sh             # 일괄 실험
    ├── configs/                # default_setting.yaml(데이터·학습), model_setting.yaml(모델)
    ├── data_provider/          # load_dataset.py(CSV 로드·정렬·분할·스케일링), build_dataset.py(윈도), factory.py
    ├── models/                 # 모델 구현과 등록(__init__.py)
    ├── layers/                 # 모델 공용 레이어
    ├── losses/, optimizers/    # 손실·옵티마이저 팩토리
    ├── utils/                  # 로그, 저장, 지표
    ├── scripts/                # solar/, wind/, Archive/ 실행 스크립트
    └── jupyter/                # 결과 시각화 노트북
```

## 🛠️ 확장

- **새 모델**: `src/models/`에 구현 → `src/models/__init__.py`에 등록 → `src/configs/model_setting.yaml`에 같은 이름으로 설정 추가
- **새 데이터셋**: [내 데이터에 맞추기](#5-내-데이터에-맞추기) 참고

## 🙏 참고 구현

- `DLinear`, `PatchTST`, `iTransformer`, `TimesNet` 등과 `src/layers/`는
  [Time-Series-Library](https://github.com/thuml/Time-Series-Library)(MIT)의 구현을 바탕으로 합니다.
- `TabM`은 [yandex-research/tabm](https://github.com/yandex-research/tabm)(Apache-2.0)을 바탕으로 합니다.

## 📝 라이선스

[MIT License](LICENSE)
