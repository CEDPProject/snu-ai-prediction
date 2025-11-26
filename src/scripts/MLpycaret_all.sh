export CUDA_VISIBLE_DEVICES=3

CATEGORY_NAMES=("solar" "wind")
MODELS=("MLpycaret")
SEQ_LENS=(1)

# 앙상블 ON/OFF 실험
ENSEMBLE_ENABLES=(True)

for ENSEMBLE_ENABLE in "${ENSEMBLE_ENABLES[@]}"; do
    echo "============================================"
    echo "Starting experiment with ENSEMBLE=$ENSEMBLE_ENABLE"
    echo "============================================"
    
    for CATEGORY_NAME in "${CATEGORY_NAMES[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            for SEQ_LEN in "${SEQ_LENS[@]}"; do
                echo "Running: MODEL=$MODEL, CATEGORY=$CATEGORY_NAME, SEQ_LEN=$SEQ_LEN, ENSEMBLE=$ENSEMBLE_ENABLE"

                python main.py \
                    --model_name "$MODEL" \
                    --default_cfg ./configs/default_setting.yaml \
                    --model_cfg ./configs/model_setting.yaml \
                    --opts "DATASET.category_name=$CATEGORY_NAME" \
                           "DATASET.seq_len=$SEQ_LEN" \
                           "MODELSETTING.selected_models=['rf', 'et', 'gbr', 'lightgbm']" \
                           "MODELSETTING.ensemble.enable=$ENSEMBLE_ENABLE"
            done
        done
    done
done


for ENSEMBLE_ENABLE in "${ENSEMBLE_ENABLES[@]}"; do
    echo "============================================"
    echo "Starting experiment with ENSEMBLE=$ENSEMBLE_ENABLE"
    echo "============================================"
    
    for CATEGORY_NAME in "${CATEGORY_NAMES[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            for SEQ_LEN in "${SEQ_LENS[@]}"; do
                echo "Running: MODEL=$MODEL, CATEGORY=$CATEGORY_NAME, SEQ_LEN=$SEQ_LEN, ENSEMBLE=$ENSEMBLE_ENABLE"

                python main.py \
                    --model_name "$MODEL" \
                    --default_cfg ./configs/default_setting.yaml \
                    --model_cfg ./configs/model_setting.yaml \
                    --opts "DATASET.category_name=$CATEGORY_NAME" \
                           "DATASET.seq_len=$SEQ_LEN" \
                           "MODELSETTING.selected_models=['rf', 'et', 'gbr', 'lightgbm']" \
                           "MODELSETTING.ensemble.enable=$ENSEMBLE_ENABLE" \
                           "MODELSETTING.search_algorithm='grid'" \
                           "MODELSETTING.enable_tuning=true"
            done
        done
    done
done



                        #    "MODELSETTING.selected_models=['svm', 'dt', 'lr', 'ridge', 'huber', 'rf', 'et', 'gbr', 'lightgbm']" \
#                            "MODELSETTING.selected_models=['rf', 'et', 'gbr', 'lightgbm']" \