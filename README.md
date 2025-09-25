# SNU-AI-Prediction

딥러닝 및 머신러닝 기반 사전에 예측된 기후 정보를 바탕으로 발전량 예측을 시스템 구현을 위한 PyTorch 프레임워크입니다. 태양광(Solar) 및 풍력(Wind) 에너지 발전량 예측에 특화되어 있습니다.

## 🚀 설치 방법

### 1. 저장소 복제
```bash
git clone https://github.com/yourusername/snu-ai-prediction.git
cd snu-ai-prediction
```

### 2. 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

## 💻 사용법

### 빠른 시작
전체 실험을 자동으로 실행:
```bash
bash run_test.sh
```

### 개별 모델 실행
특정 모델과 설정으로 실행:
```bash
python main.py \
    --model_name DLinear \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    --opts "DATASET.category_name=solar" "DATASET.seq_len=24"
```

### 머신러닝 앙상블 실행
```bash
python main.py \
    --model_name MLpycaret \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    --opts "DATASET.category_name=solar" \
           "DATASET.seq_len=1" \
           "MODELSETTING.ensemble.enable=True"
```

## 🧪 실험 설정

### 지원 모델
#### 딥러닝 모델
- **DLinear**
- **PatchTST**
- **iTransformer**
- **TimesNet**
- **NCART**

#### 머신러닝 모델 (PyCaret 앙상블)
- SVM, Decision Tree, Linear Regression
- Ridge, Huber, Random Forest
- Extra Trees, Gradient Boosting, LightGBM

### 데이터셋
- **Solar**: 태양광 발전 데이터
- **Wind**: 풍력 발전 데이터

### 예측 구간 (Sequence Length)
- 1시간 (seq_len=1)
- 12시간 (seq_len=12)
- 24시간 (seq_len=24)
- 임의대로 설정 가능

## 📁 프로젝트 구조

```
snu-ai-prediction/
│
├── src/
│   ├── __pycache__/
│   │
│   ├── configs/
│   │   ├── default_setting.yaml
│   │   └── model_setting.yaml
│   │
│   ├── data_provider/
│   │   ├── __init__.py
│   │   ├── build_dataset.py
│   │   ├── factory.py
│   │   └── load_dataset.py
│   │
│   ├── layers/
│   │   ├── __pycache__/
│   │   ├── Conv_Blocks.py
│   │   ├── decomposition.py
│   │   ├── Embed.py
│   │   ├── SelfAttention_Family.py
│   │   └── Transformer_EncDec.py
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   └── factory.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── DLinear.py
│   │   ├── factory.py
│   │   ├── iTransformer.py
│   │   ├── Koopa.py
│   │   ├── MICN.py
│   │   ├── MLModelEnsemble.py
│   │   ├── NCART.py
│   │   ├── PatchTST.py
│   │   ├── TabM.py
│   │   ├── TiDE.py
│   │   └── TimesNet.py
│   │
│   ├── optimizers/
│   │   ├── __init__.py
│   │   └── factory.py
│   │
│   ├── scripts/
│   │
│   └── utils/
│       ├── savefile/
│       ├── log.py
│       ├── masking.py
│       ├── tools.py
│       └── utils.py
│
├── arguments.py
├── exp_builder_dl.py
├── exp_builder_ml.py
├── main.py
├── run_test.sh
├── test.ipynb
├── logs.log
├── requirements.txt
├── LICENSE
├── README.md
└── .gitignore
```

## 🛠️ 개발

### 새로운 모델 추가하기
1. `src/models/`에 모델 구현
2. `src/models/__init__.py`에 등록
3. `configs/model_setting`에 해당 모델 설정 파일 생성

### 새로운 데이터셋 추가하기
1. `src/data_provider/load_dataset.py`에 데이터 로더 구현
2. 필요에 따라 설정 파일 업데이트

## 📝 라이선스

이 프로젝트는 [LICENSE](LICENSE) 파일에 명시된 조건에 따라 라이선스가 부여됩니다.

---
**참고**: 이 프로젝트는 활발히 개발 중입니다. 정기적으로 업데이트를 확인해주세요.