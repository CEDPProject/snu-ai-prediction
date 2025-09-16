export CUDA_VISIBLE_DEVICES=3

#!/bin/bash

# 기존 변수들
ENSEMBLE_ENABLES=(True)
CATEGORY_NAMES=("solar" "wind")
MODELS=("MLpycaret")
SEQ_LENS=(1)

# ignore_features 조합 추가
declare -a IGNORE_FEATURES_SETS=(
    "[]"
    "['LEAD_TM', 'NWP_TP_CD']"
    "['LEAD_TM']"
    "['NWP_TP_CD']"
)

# 실험 실행
for ENSEMBLE_ENABLE in "${ENSEMBLE_ENABLES[@]}"; do
    echo "============================================"
    echo "Starting experiment with ENSEMBLE=$ENSEMBLE_ENABLE"
    echo "============================================"
    
    for CATEGORY_NAME in "${CATEGORY_NAMES[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            for SEQ_LEN in "${SEQ_LENS[@]}"; do
                for IGNORE_FEATURES in "${IGNORE_FEATURES_SETS[@]}"; do
                    echo "Running: MODEL=$MODEL, CATEGORY=$CATEGORY_NAME, SEQ_LEN=$SEQ_LEN, ENSEMBLE=$ENSEMBLE_ENABLE, IGNORE_FEATURES=$IGNORE_FEATURES"

                    python main.py \
                        --model_name "$MODEL" \
                        --default_cfg ./configs/default_setting.yaml \
                        --model_cfg ./configs/model_setting.yaml \
                        --opts "DATASET.category_name=$CATEGORY_NAME" \
                               "DATASET.seq_len=$SEQ_LEN" \
                               "MODELSETTING.selected_models=['svm', 'dt', 'lr', 'ridge', 'huber', 'rf', 'et', 'gbr', 'lightgbm']" \
                               "MODELSETTING.ensemble.enable=$ENSEMBLE_ENABLE" \
                               "DATASET.ignore_feature=$IGNORE_FEATURES"
                done
            done
        done
    done
done

