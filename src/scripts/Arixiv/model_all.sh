export CUDA_VISIBLE_DEVICES=2

TARGET_TYPES=("multi")
CATEGORY_NAMES=("solar" "wind")

ENSEMBLE_ENABLES=(True)
ENSEMBLE_TYPES_LISTS=(
    "['bagging']"
)
MODELS=("MLpycaret")
SEQ_LENS=(1)
IGNORE_FEATURES_SETS=(
    "['NWP_TP_CD']"
)

for TARGET_TYPE in "${TARGET_TYPES[@]}"; do
    for ENSEMBLE_ENABLE in "${ENSEMBLE_ENABLES[@]}"; do
        echo "============================================"
        echo "Start: ENSEMBLE_ENABLE=$ENSEMBLE_ENABLE | TARGET_TYPE=$TARGET_TYPE"
        echo "============================================"

        for ENSEMBLE_TYPES in "${ENSEMBLE_TYPES_LISTS[@]}"; do
            for CATEGORY_NAME in "${CATEGORY_NAMES[@]}"; do
                for MODEL in "${MODELS[@]}"; do
                    for SEQ_LEN in "${SEQ_LENS[@]}"; do
                        for IGNORE_FEATURES in "${IGNORE_FEATURES_SETS[@]}"; do
                            if [ "$TARGET_TYPE" == "multi" ]; then
                                PRED_LEN=$SEQ_LEN
                            else
                                PRED_LEN=1
                            fi

                            echo "MODEL=$MODEL | CAT=$CATEGORY_NAME | SEQ_LEN=$SEQ_LEN | TARGET_TYPE=$TARGET_TYPE"
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
                                    "DATASET.pred_len=$PRED_LEN" \
                                    "DATASET.target_type=$TARGET_TYPE" \
                                    "DATASET.split_rate=[1.0, 0.0, 0.0]" \
                                    "DATASET.split_date='2023-01-01 00:00:00'" \
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
done


# MODELS=("DLinear" "PatchTST")
# SEQ_LENS=(1 12 24)

# for TARGET_TYPE in "${TARGET_TYPES[@]}"; do
#     for CATEGORY_NAME in "${CATEGORY_NAMES[@]}"; do
#         for MODEL in "${MODELS[@]}"; do
#             for SEQ_LEN in "${SEQ_LENS[@]}"; do
#                 if [ "$TARGET_TYPE" == "multi" ]; then
#                     PRED_LEN=$SEQ_LEN
#                 else
#                     PRED_LEN=1
#                 fi

#                 echo "Running MODEL=$MODEL | CAT=$CATEGORY_NAME | SEQ_LEN=$SEQ_LEN | TARGET_TYPE=$TARGET_TYPE | PRED_LEN=$PRED_LEN"

#                 python main.py \
#                     --model_name "$MODEL" \
#                     --default_cfg ./configs/default_setting.yaml \
#                     --model_cfg ./configs/model_setting.yaml \
#                     --opts \
#                         "DATASET.category_name=$CATEGORY_NAME" \
#                         "DATASET.seq_len=$SEQ_LEN" \
#                         "DATASET.pred_len=$PRED_LEN" \
#                         "DATASET.target_type=$TARGET_TYPE" \
#                         "DATASET.split_rate=[0.95, 0.05, 0.0]" \
#                         "DATASET.split_date='2023-01-01 00:00:00'"
#             done
#         done
#     done
# done


###############################
# 세 번째 실험 블록 (PyCaret 기반 앙상블)
###############################



# ###############################
# # 첫 번째 실험 블록 (기본 모델들)
# ###############################

# MODELS=("DLinear")
# SEQ_LENS=(12 24)

# for TARGET_TYPE in "${TARGET_TYPES[@]}"; do
#     for CATEGORY_NAME in "${CATEGORY_NAMES[@]}"; do
#         for MODEL in "${MODELS[@]}"; do
#             for SEQ_LEN in "${SEQ_LENS[@]}"; do
#                 if [ "$TARGET_TYPE" == "multi" ]; then
#                     PRED_LEN=$SEQ_LEN
#                 else
#                     PRED_LEN=1
#                 fi

#                 echo "Running MODEL=$MODEL | CAT=$CATEGORY_NAME | SEQ_LEN=$SEQ_LEN | TARGET_TYPE=$TARGET_TYPE | PRED_LEN=$PRED_LEN"

#                 python main.py \
#                     --model_name "$MODEL" \
#                     --default_cfg ./configs/default_setting.yaml \
#                     --model_cfg ./configs/model_setting.yaml \
#                     --opts \
#                         "DATASET.category_name=$CATEGORY_NAME" \
#                         "DATASET.seq_len=$SEQ_LEN" \
#                         "DATASET.pred_len=$PRED_LEN" \
#                         "DATASET.target_type=$TARGET_TYPE" \
#                         "DATASET.split_rate=[0.95, 0.05, 0.0]" \
#                         "DATASET.split_date='2023-01-01 00:00:00'"
#             done
#         done
#     done
# done


# ###############################
# # 두 번째 실험 블록 (TabM, NCART)
# ###############################

# MODELS=("TabM" "NCART")

# for TARGET_TYPE in "${TARGET_TYPES[@]}"; do
#     for CATEGORY_NAME in "${CATEGORY_NAMES[@]}"; do
#         for MODEL in "${MODELS[@]}"; do
#             for SEQ_LEN in "${SEQ_LENS[@]}"; do
#                 if [ "$TARGET_TYPE" == "multi" ]; then
#                     PRED_LEN=$SEQ_LEN
#                 else
#                     PRED_LEN=1
#                 fi

#                 echo "Running MODEL=$MODEL | CAT=$CATEGORY_NAME | SEQ_LEN=$SEQ_LEN | TARGET_TYPE=$TARGET_TYPE | PRED_LEN=$PRED_LEN"

#                 python main.py \
#                     --model_name "$MODEL" \
#                     --default_cfg ./configs/default_setting.yaml \
#                     --model_cfg ./configs/model_setting.yaml \
#                     --opts \
#                         "DATASET.category_name=$CATEGORY_NAME" \
#                         "DATASET.seq_len=$SEQ_LEN" \
#                         "DATASET.pred_len=$PRED_LEN" \
#                         "DATASET.target_type=$TARGET_TYPE" \
#                         "DATASET.split_rate=[0.95, 0.05, 0.0]" \
#                         "DATASET.split_date='2023-01-01 00:00:00'"
#             done
#         done
#     done
# done
