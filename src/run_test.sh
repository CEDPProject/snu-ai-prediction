export CUDA_VISIBLE_DEVICES=0
python main.py \
    --model_name MLModelEnsemble \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    TRAIN.fine_tuning_num 1
