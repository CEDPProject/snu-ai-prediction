#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

# 1. 실험 변수 -----------------------------------------------------------
ENSEMBLE_ENABLES=(True)
ENSEMBLE_TYPES_LISTS=(
    "['bagging']"
)

CATEGORY_NAMES=("solar" "wind")
MODELS=("MLpycaret")
SEQ_LENS=(1)

# ignore_features 조합
IGNORE_FEATURES_SETS=(
    "['NWP_TP_CD']"
)
# -----------------------------------------------------------------------

for ENSEMBLE_ENABLE in "${ENSEMBLE_ENABLES[@]}"; do
    echo "============================================"
    echo "Start: ENSEMBLE_ENABLE=$ENSEMBLE_ENABLE"
    echo "============================================"

    for ENSEMBLE_TYPES in "${ENSEMBLE_TYPES_LISTS[@]}"; do
        for CATEGORY_NAME in "${CATEGORY_NAMES[@]}"; do
            for MODEL in "${MODELS[@]}"; do
                for SEQ_LEN in "${SEQ_LENS[@]}"; do
                    for IGNORE_FEATURES in "${IGNORE_FEATURES_SETS[@]}"; do
                        echo "MODEL=$MODEL | CAT=$CATEGORY_NAME | SEQ_LEN=$SEQ_LEN"
                        echo "ENSEMBLE_ENABLE=$ENSEMBLE_ENABLE | ENSEMBLE_TYPES=$ENSEMBLE_TYPES"
                        echo "IGNORE_FEATURES=$IGNORE_FEATURES"
                        echo "------------------------------------------------------------"

                        python main.py \
                            --model_name "$MODEL" \
                            --default_cfg ./configs/default_setting.yaml \
                            --model_cfg   ./configs/model_setting.yaml \
                            --opts \
                                "DATASET.category_name=$CATEGORY_NAME" \
                                "DATASET.seq_len=$SEQ_LEN" \
                                "MODELSETTING.selected_models=['rf', 'et']" \
                                "MODELSETTING.ensemble.enable=$ENSEMBLE_ENABLE" \
                                "MODELSETTING.ensemble.types=$ENSEMBLE_TYPES" \
                                "DATASET.ignore_feature=$IGNORE_FEATURES"
                    done
                done
            done
        done
    done
done

    # "['blend']"
    # "['stack']"
    # "['bagging']"