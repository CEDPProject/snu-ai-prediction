export CUDA_VISIBLE_DEVICES=3

CATEGORY_NAMES=("solar" "wind")
MODELS=("TabM" "NCART")
SEQ_LENS=(1)

for CATEGORY_NAME in "${CATEGORY_NAMES[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for SEQ_LEN in "${SEQ_LENS[@]}"; do
            echo "Running experiment with MODEL=$MODEL, CATEGORY_NAME=$CATEGORY_NAME, SEQ_LEN=$SEQ_LEN"

            python main.py \
                --model_name "$MODEL" \
                --default_cfg ./configs/default_setting.yaml \
                --model_cfg ./configs/model_setting.yaml \
                --opts "DATASET.category_name=$CATEGORY_NAME" \
                       "DATASET.seq_len=$SEQ_LEN" \
                       "DATASET.split_rate=[0.8,0.1,0.1]" \
                       "DATASET.split_date=null"
        done
    done
done
