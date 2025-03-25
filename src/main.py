import numpy as np
import os
import wandb
import json
import shutil
import logging
from glob import glob
import torch
from omegaconf import OmegaConf

from exp_builder_dl import training_dl, test_dl
from exp_builder_ml import training_ml, test_ml, training_ml_pycaret, test_ml_pycaret
from data_provider import create_dataloader
from losses import create_criterion
from optimizers import create_optimizer
from models import create_model
from utils.log import setup_default_logging
from utils.utils import version_build, load_resume_model, Float32Encoder
from utils.tools import update_information
from arguments import parser

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

# set logger
_logger = get_logger('train')

def main(cfg):
    # set logger setting
    setup_default_logging()
    
    # set seed
    set_seed(cfg.DEFAULT.seed)
    
    # set accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps = cfg.TRAIN.grad_accum_steps,
        mixed_precision             = cfg.TRAIN.mixed_precision
    )

    # make save directory
    savedir = os.path.join(cfg.RESULT.savedir,
                            cfg.DATASET.data_version, 
                            cfg.MODEL.modelname, 
                            cfg.DEFAULT.exp_name)
    
    loaddir, savedir = version_build(accelerator = accelerator,
                                    logdir = savedir, 
                                    resume = cfg.TRAIN.resume)
    
    # set device
    if accelerator.is_local_main_process:
        _logger.info('Device: {}'.format(accelerator.device))
    
    # load and define dataloader
    trn_dataloader, val_dataloader, tst_dataloader, features, scaler_dict = create_dataloader(
                                                datadir                  = cfg.DATASET.datadir,
                                                scale                    = cfg.DATASET.scale,
                                                batch_size               = cfg.DATASET.batch_size,
                                                shuffle                  = cfg.DATASET.shuffle,
                                                num_workers              = cfg.DATASET.num_workers,
                                                pin_memory               = cfg.DATASET.pin_memory,
                                                pre_training             = cfg.TRAIN.pre_training,
                                                drop_last                = cfg.DATASET.drop_last,
                                                category_name            = cfg.DATASET.category_name,
                                                input_structure          = cfg.DATASET.input_structure,
                                                data_version             = cfg.DATASET.data_version,
                                                lead_time_align          = cfg.DATASET.lead_time_align,
                                                split_rate               = cfg.DATASET.split_rate,
                                                seq_len                  = cfg.DATASET.seq_len,
                                                pred_len                 = cfg.DATASET.pred_len,
                                                align_time               = cfg.DATASET.align_time,
                                                target                   = cfg.DATASET.target,
                                                model_name               = cfg.MODEL.modelname,
                                                del_feature              = cfg.DATASET.ignore_feature
                                                )
    
    # update cfg
    update_information(cfg       =  cfg, 
                        section  =  'MODELSETTING',
                        dim_in   =  len(features),
                        seq_len  =  cfg.DATASET.seq_len,
                        pred_len =  cfg.DATASET.pred_len,
                        c_out    =  cfg.DATASET.c_out,
                        )
    
    # save configs
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        OmegaConf.save(cfg, os.path.join(savedir, 'configs.yaml'))
        print(OmegaConf.to_yaml(cfg))

    # wandb
    if cfg.TRAIN.wandb.use:
        # initialize wandb
        wandb.init(name    = cfg.DEFAULT.exp_name, 
                    group   = cfg.TRAIN.wandb.exp_name,
                    project = cfg.TRAIN.wandb.project_name, 
                    entity  = cfg.TRAIN.wandb.entity, 
                    config  = OmegaConf.to_container(cfg))
    
    criterion = create_criterion(loss_name=cfg.LOSS.loss_name)
    

    if 'ML' in cfg.MODEL.modelname:
        if cfg.MODEL.modelname != 'MLpycaret':
            # build Model
            model = create_model(
                modelname    = cfg.MODEL.modelname,
                params       = cfg.MODELSETTING
                )
            
            _logger.info('Model name: {}'.format(cfg.MODEL.modelname))

        # ML Model Training
        _logger.info('Training ML Model...')
        if cfg.MODEL.modelname == 'MLpycaret':
            all_trained_models_dict, feature_names_dict = training_ml_pycaret(
                                    trn_dataloader,
                                    selected_models         = cfg.MODELSETTING.selected_models,
                                    save_dir                = savedir,
                                    enable_tuning           = cfg.MODELSETTING.enable_tuning,
                                    tuning_iter             = cfg.MODELSETTING.tuning_iter,
                                    optimize_metric         = cfg.MODELSETTING.optimize_metric,
                                    search_library          = cfg.MODELSETTING.search_library,
                                    search_algorithm        = cfg.MODELSETTING.search_algorithm,
                                    early_stopping          = cfg.MODELSETTING.early_stopping,
                                    n_jobs                  = cfg.MODELSETTING.n_jobs,
                                    grid_model              = cfg.MODELSETTING.model_grids,
                                    k_fold_num              = cfg.MODELSETTING.k_fold_num,
                                    category_name           = cfg.DATASET.category_name,
                                    ignore_features_list    = cfg.DATASET.ignore_feature
                                    )
            
            test_metrics = test_ml_pycaret(all_trained_models_dict, 
                                    tst_dataloader, 
                                    feature_names_dict,
                                    save_dir = savedir,
                                    eval_merge_grp = True, 
                                    category_name = cfg.DATASET.category_name)
            _logger.info('Pycaret done!')
            exit()
        else:
            model_dict = training_ml(model, 
                                    trn_dataloader,
                                    eval_merge_grp = True, 
                                    category_name = cfg.DATASET.category_name)
            
            test_metrics = test_ml(model_dict, 
                                    tst_dataloader, 
                                    eval_merge_grp = True, 
                                    category_name = cfg.DATASET.category_name)
    else:
        # 모델 및 옵티마이저 생성 부분 (main 코드)
        if cfg.DATASET.individual_model:
            # 각 호기(데이터로더의 키)에 대해 개별 모델 생성
            models = {}
            optimizers = {}
            for key in trn_dataloader.keys():
                model_instance = create_model(
                    modelname=cfg.MODEL.modelname,
                    params=cfg.MODELSETTING
                )
                _logger.info(f"{key} for Model Name: {cfg.MODEL.modelname}")
                _logger.info(f"# of learnable params for {key}: {np.sum([p.numel() if p.requires_grad else 0 for p in model_instance.parameters()])}")
                if cfg.TRAIN.resume is not None:
                    load_resume_model(
                        model=model_instance, 
                        loaddir=loaddir, 
                        resume=cfg.TRAIN.resume, 
                        pre_training=cfg.TRAIN.pre_training,
                        fine_tuning_method=cfg.TRAIN.fine_tuning_method
                    )
                optimizer_instance = create_optimizer(
                    model=model_instance, 
                    opt_name=cfg.OPTIMIZER.opt_name, 
                    lr=cfg.OPTIMIZER.lr, 
                    params=cfg.OPTIMIZER.params
                )
                models[key] = model_instance
                optimizers[key] = optimizer_instance
            # accelerator로 각각 준비
            for key in models:
                models[key], optimizers[key] = accelerator.prepare(models[key], optimizers[key])
        else:
            model = create_model(
                modelname=cfg.MODEL.modelname,
                params=cfg.MODELSETTING
            )
            _logger.info('Model name: {}'.format(cfg.MODEL.modelname))
            if cfg.TRAIN.resume is not None:
                load_resume_model(
                    model=model, 
                    loaddir=loaddir, 
                    resume=cfg.TRAIN.resume, 
                    pre_training=cfg.TRAIN.pre_training,
                    fine_tuning_method=cfg.TRAIN.fine_tuning_method
                )
            _logger.info('# of learnable params: {}'.format(np.sum([p.numel() if p.requires_grad else 0 for p in model.parameters()])))
            optimizer = create_optimizer(
                model=model, 
                opt_name=cfg.OPTIMIZER.opt_name, 
                lr=cfg.OPTIMIZER.lr, 
                params=cfg.OPTIMIZER.params
            )
            model, optimizer = accelerator.prepare(model, optimizer)

        # DataLoader들도 accelerator로 준비
        for key in trn_dataloader:
            trn_dataloader[key] = accelerator.prepare(trn_dataloader[key])
        for key in val_dataloader:
            val_dataloader[key] = accelerator.prepare(val_dataloader[key])
        for key in tst_dataloader:
            tst_dataloader[key] = accelerator.prepare(tst_dataloader[key])

        # DL 모델 학습 실행 (individual_model에 따라 분기)
        _logger.info('Training DL Model...')

        if cfg.DATASET.individual_model:
            training_dl(
                model                = models,
                trainloader          = trn_dataloader,
                validloader          = val_dataloader,
                criterion            = criterion,
                optimizer            = optimizers,
                accelerator          = accelerator, 
                epochs               = cfg.TRAIN.epoch,
                eval_epochs          = cfg.TRAIN.eval_epochs, 
                log_epochs           = cfg.TRAIN.log_epochs,
                log_eval_iter        = cfg.TRAIN.log_eval_iter,
                use_wandb            = cfg.TRAIN.wandb.use, 
                wandb_iter           = cfg.TRAIN.wandb.wandb_iter, 
                ckp_metric           = cfg.TRAIN.ckp_metric, 
                savedir              = savedir,
                model_name           = cfg.MODEL.modelname,
                early_stopping_count = cfg.TRAIN.early_stopping_count,
                pre_training         = cfg.TRAIN.pre_training,
                return_output        = False
            )
        else:
            training_dl(
                model                = model, 
                trainloader          = trn_dataloader, 
                validloader          = val_dataloader,
                criterion            = criterion, 
                optimizer            = optimizer,
                accelerator          = accelerator, 
                epochs               = cfg.TRAIN.epoch,
                eval_epochs          = cfg.TRAIN.eval_epochs, 
                log_epochs           = cfg.TRAIN.log_epochs,
                log_eval_iter        = cfg.TRAIN.log_eval_iter,
                use_wandb            = cfg.TRAIN.wandb.use, 
                wandb_iter           = cfg.TRAIN.wandb.wandb_iter, 
                ckp_metric           = cfg.TRAIN.ckp_metric, 
                savedir              = savedir,
                model_name           = cfg.MODEL.modelname,
                early_stopping_count = cfg.TRAIN.early_stopping_count,
                pre_training         = cfg.TRAIN.pre_training,
                return_output        = False
            )

        # 최적 모델 로드 (개별 모델일 경우 각 호기별로 로드)
        if cfg.DATASET.individual_model:
            for key in models:
                model_path = os.path.join(savedir, f'best_model_{key[0]}_{key[1]}.pt')
                models[key].load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(os.path.join(savedir, 'best_model.pt')))

        # VALID, TEST 평가도 individual_model에 따라 분기
        if cfg.DATASET.individual_model:
            # val_metrics = test_dl(
            #     model          = models, 
            #     dataloader     = val_dataloader, 
            #     criterion      = criterion,
            #     log_eval_iter  = cfg.TRAIN.log_eval_iter,
            #     name           = 'VALID',
            #     pre_training   = cfg.TRAIN.pre_training,
            #     savedir        = savedir,
            #     accelerator    = accelerator,
            #     model_name     = cfg.MODEL.modelname,
            #     return_output  = False
            # )
            
            test_metrics = test_dl(
                model          = models, 
                dataloader     = tst_dataloader, 
                criterion      = criterion,
                log_eval_iter  = cfg.TRAIN.log_eval_iter,
                name           = 'TEST',
                pre_training   = cfg.TRAIN.pre_training,
                savedir        = savedir,
                accelerator    = accelerator,
                model_name     = cfg.MODEL.modelname,
                return_output  = True
            )
        else:
            val_metrics = test_dl(
                model          = model, 
                dataloader     = val_dataloader, 
                criterion      = criterion,
                log_eval_iter  = cfg.TRAIN.log_eval_iter,
                name           = 'VALID',
                pre_training   = cfg.TRAIN.pre_training,
                savedir        = savedir,
                accelerator    = accelerator,
                model_name     = cfg.MODEL.modelname,
                return_output  = False
            )
            
            test_metrics = test_dl(
                model          = model, 
                dataloader     = tst_dataloader, 
                criterion      = criterion,
                log_eval_iter  = cfg.TRAIN.log_eval_iter,
                name           = 'TEST',
                pre_training   = cfg.TRAIN.pre_training,
                savedir        = savedir,
                accelerator    = accelerator,
                model_name     = cfg.MODEL.modelname,
                return_output  = True
            )

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            test_metrics = {str(key): value for key, value in test_metrics.items()}
            _logger.info('test_metrics: {}'.format(test_metrics))
            json.dump(test_metrics, open(os.path.join(savedir, 'test_results.json'),'w'), indent='\t', cls=Float32Encoder)

            # 모든 작업 완료 후 .pt 파일 삭제
            if cfg.DEFAULT.del_pt:
                import glob
                pt_files = glob.glob(os.path.join(savedir, "*.pt"))
                for pt_file in pt_files:
                    try:
                        os.remove(pt_file)
                        print(f"[INFO] Deleted: {pt_file}")
                    except Exception as e:
                        print(f"[WARNING] Failed to delete {pt_file}: {e}")

if __name__=='__main__':
    cfg = parser()
    main(cfg)
