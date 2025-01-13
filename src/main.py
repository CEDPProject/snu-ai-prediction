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
                                                align_time               = cfg.DATASET.align_time,
                                                target                   = cfg.DATASET.target,
                                                model_name               = cfg.MODEL.modelname,
                                                )
    
    # update cfg
    update_information(cfg      = cfg, 
                        section = 'MODELSETTING',
                        key     = 'dim_in',
                        value   = len(features))
    
    # save configs
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        OmegaConf.save(cfg, os.path.join(savedir, 'configs.yaml'))
        print(OmegaConf.to_yaml(cfg))
    
    if cfg.MODEL.modelname != 'MLpycaret':
        # build Model
        model = create_model(
            modelname    = cfg.MODEL.modelname,
            params       = cfg.MODELSETTING
            )
        
        _logger.info('Model name: {}'.format(model))
    
    # wandb
    if cfg.TRAIN.wandb.use:
        # initialize wandb
        wandb.init(name    = cfg.DEFAULT.exp_name, 
                   group   = cfg.TRAIN.wandb.exp_name,
                   project = cfg.TRAIN.wandb.project_name, 
                   entity  = cfg.TRAIN.wandb.entity, 
                   config  = OmegaConf.to_container(cfg))
        
    if 'ML' in cfg.MODEL.modelname:
        # ML Model Training
        _logger.info('Training ML Model...')
        if cfg.MODEL.modelname == 'MLpycaret':
            all_trained_models_dict, feature_names_dict = training_ml_pycaret(
                                    trn_dataloader,
                                    selected_models = cfg.MODELSETTING.selected_models,
                                    save_dir = savedir,
                                    enable_tuning=cfg.MODELSETTING.enable_tuning,
                                    tuning_iter=cfg.MODELSETTING.tuning_iter,
                                    optimize_metric=cfg.MODELSETTING.optimize_metric,
                                    search_library=cfg.MODELSETTING.search_library,
                                    search_algorithm=cfg.MODELSETTING.search_algorithm,
                                    early_stopping=cfg.MODELSETTING.early_stopping,
                                    n_jobs=cfg.MODELSETTING.n_jobs,
                                    grid_model=cfg.MODELSETTING.model_grids,
                                    k_fold_num=cfg.MODELSETTING.k_fold_num,
                                    category_name = cfg.DATASET.category_name,
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
        # load weights
        if cfg.TRAIN.resume is not None:
            load_resume_model(model=model, 
                            loaddir=loaddir, 
                            resume=cfg.TRAIN.resume, 
                            pre_training=cfg.TRAIN.pre_training,
                            fine_tuning_method=cfg.TRAIN.fine_tuning_method)
        
        _logger.info('# of learnable params: {}'.format(np.sum([p.numel() if p.requires_grad else 0 for p in model.parameters()])))
        
        model, optimizer, trn_dataloader, val_dataloader, tst_dataloader = accelerator.prepare(
            model, optimizer, trn_dataloader, val_dataloader, tst_dataloader
        )
        
        # set training
        criterion = create_criterion(loss_name=cfg.LOSS.loss_name)
        
        # set optimizer
        optimizer = create_optimizer(model=model, opt_name=cfg.OPTIMIZER.opt_name, lr=cfg.OPTIMIZER.lr, params=cfg.OPTIMIZER.params)

        # DL Model Training
        _logger.info('Training DL Model...')
        
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
        classification       = cfg.DATASET.classification,
        return_output        = False
        )
        
        # load best checkpoint weights
        model.load_state_dict(torch.load(os.path.join(savedir, 'best_model.pt')))

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
        classification = cfg.DATASET.classification, 
        return_output  = False
        )
        
        # Test DL Model
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
        classification = cfg.DATASET.classification, 
        return_output  = False
        )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        test_metrics = {str(key): value for key, value in test_metrics.items()}
        _logger.info('test_metrics: {}'.format(test_metrics))
        json.dump(test_metrics, open(os.path.join(savedir, 
                                f'test_results.json'),'w'), indent='\t', cls=Float32Encoder)

if __name__=='__main__':
    cfg = parser()
    main(cfg)
