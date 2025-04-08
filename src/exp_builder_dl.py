import logging
import wandb
import time
import os
import json
import numpy as np
import torch
from collections import OrderedDict
from accelerate import Accelerator
from utils.utils import cal_metric, Float32Encoder
from utils.tools import EarlyStopping

_logger = logging.getLogger('train')

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def training_dl(model, trainloader, validloader, criterion, optimizer, accelerator: Accelerator, 
                epochs: int, eval_epochs: int, log_epochs: int, log_eval_iter: int, wandb_iter: int,
                use_wandb: bool, ckp_metric: str, savedir: str, model_name: str, clamping: bool, 
                pre_training: int, early_stopping_count: int, return_output: bool):

    early_stopping = EarlyStopping(patience=early_stopping_count)
    best_score = np.inf

    for (model_key, area_grp_id), loader in trainloader.items():
        _logger.info(f"🔹 Training on {model_key} - {area_grp_id}")

        # 개별 모델인 경우 딕셔너리에서 선택, 단일 모델이면 동일 모델 사용
        if isinstance(model, dict):
            current_model = model[(model_key, area_grp_id)]
            current_optimizer = optimizer[(model_key, area_grp_id)]
        else:
            current_model = model
            current_optimizer = optimizer

        batch_time_m = AverageMeter()
        data_time_m = AverageMeter()
        losses_m = AverageMeter()

        current_model.train()
        current_optimizer.zero_grad()
        end_time = time.time()

        early_stopping = EarlyStopping(patience=early_stopping_count)
        best_score = np.inf
        wandb_iteration = 0

        for epoch in range(epochs):
            for idx, item in enumerate(loader):
                data_time_m.update(time.time() - end_time)
                
                inputs = item['inputs'].float()
                targets = item['targets'].float()
                outputs = current_model(inputs)
                
                if clamping:
                    outputs = torch.clamp(outputs, min=0)
                
                if model_name == 'TabM':
                    k = outputs.shape[-1]
                    outputs = outputs.flatten(0, 1)
                    targets = targets.repeat_interleave(k)
                
                loss = criterion(outputs, targets)

                accelerator.backward(loss)
                current_optimizer.step()
                current_optimizer.zero_grad()
                
                batch_time_m.update(time.time() - end_time)
                wandb_iteration += 1

                if use_wandb and (wandb_iteration+1) % wandb_iter:
                    train_results = OrderedDict([
                        ('lr', current_optimizer.param_groups[0]['lr']),
                        ('train_loss', losses_m.avg)
                    ])
                    wandb.log(train_results, step=idx+1)
            
            if (epoch+1) % log_epochs == 0:
                _logger.info('Epoch [{epoch}/{total_epoch}] TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                            'LR: {lr:.3e} '
                            'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                            'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            (idx+1), 
                            len(loader), 
                            total_epoch = epochs,
                            epoch       = epoch + 1,
                            loss        = losses_m, 
                            lr          = current_optimizer.param_groups[0]['lr'],
                            batch_time  = batch_time_m,
                            rate        = inputs.size(0) / batch_time_m.val if batch_time_m.val > 0 else 0,
                            rate_avg    = inputs.size(0) / batch_time_m.avg if batch_time_m.avg > 0 else 0,
                            data_time   = data_time_m))

            if (epoch + 1) % eval_epochs == 0:
                if (model_key, area_grp_id) in validloader:
                    eval_metrics = test_dl(
                        model                     = current_model if not isinstance(model, dict) else { (model_key, area_grp_id): current_model },
                        dataloader                = { (model_key, area_grp_id): validloader[(model_key, area_grp_id)] },
                        criterion                 = criterion,
                        name                      = 'VALID',
                        log_eval_iter             = log_eval_iter,
                        pre_training              = pre_training,
                        savedir                   = savedir,
                        model_name                = model_name,
                        clamping                  = clamping,
                        accelerator               = accelerator,
                        return_output             = False,
                        return_avg                = True,
                        return_individual_outputs = False
                    )

                    current_model.train()
                    
                    eval_results = {f'eval_{k}': v for k, v in eval_metrics.items()}

                    if use_wandb:
                        wandb.log(eval_results, step=idx+1)

                    if best_score > eval_metrics[ckp_metric]:
                        state = {'best_epoch': epoch, 'best_step': idx+1, f'best_{ckp_metric}': eval_metrics[ckp_metric]}
                        
                        print('Save best model complete, epoch: {0}: Best metric has changed from {1:.5f} to {2:.5f}'.format(
                            epoch, best_score, eval_metrics[ckp_metric]))
            
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            state.update(eval_results)
                            # 단일 모델인 경우 파일명 지정
                            if isinstance(model, dict):
                                result_file = os.path.join(savedir, f'best_results_{model_key}_{area_grp_id}.json')
                                model_file = os.path.join(savedir, f'best_model_{model_key}_{area_grp_id}.pt')
                            else:
                                result_file = os.path.join(savedir, 'best_results.json')
                                model_file = os.path.join(savedir, 'best_model.pt')
                            json.dump(state, open(result_file, 'w'), indent='\t', cls=Float32Encoder)
                            torch.save(current_model.state_dict(), model_file)
                            if isinstance(model, dict):
                                _logger.info(f"✅ Saved best model for {model_key} - {area_grp_id}")
                            else:
                                _logger.info("✅ Saved best model")
                        _logger.info('Best {0} {1:6.6f} to {2:6.6f}'.format(ckp_metric.upper(), best_score, eval_metrics[ckp_metric]))

                        best_score = eval_metrics[ckp_metric]

                    early_stopping(eval_metrics['loss'])
                    if early_stopping.early_stop:
                        _logger.info("⏳ Early stopping triggered")
                        break
                else:
                    _logger.info(f"🚨 No validation data for {model_key} - {area_grp_id}, skipping validation.")

            end_time = time.time()

    _logger.info("🎉 Training complete for all datasets")


def test_dl(model, dataloader, criterion, accelerator: Accelerator, pre_training: int, log_eval_iter: int, 
            savedir: str, model_name: str, clamping: bool, name: str = 'TEST', return_output: bool = True, return_avg: bool = True, 
            return_individual_outputs: bool = False):
    
    _logger.info(f'\n[🔍 Start {name} Evaluation]')

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    
    total_targets = {}
    total_outputs = {}

    end_time = time.time()
    wandb_iteration = 0

    for (model_key, area_grp_id), loader in dataloader.items():
        _logger.info(f"🧪 Testing on {model_key} - {area_grp_id}")
        outputs_list, targets_list = [], []
        if isinstance(model, dict):
            current_model = model[(model_key, area_grp_id)]
        else:
            current_model = model
        for idx, item in enumerate(loader):
            data_time_m.update(time.time() - end_time)

            inputs = item['inputs'].float()
            targets = item['targets'].float()
                
            outputs = current_model(inputs)

            if clamping:
                outputs = torch.clamp(outputs, min=0)
        
            if model_name == 'TabM':
                k = outputs.shape[-1]
                outputs = outputs.flatten(0, 1)
                targets = targets.repeat_interleave(k)
                
            loss = criterion(outputs, targets)
            loss = accelerator.gather(loss).mean()

            
            outputs, targets = accelerator.gather_for_metrics((outputs.contiguous(), targets.contiguous()))
            losses_m.update(loss.item(), n=targets.size(0))
            
            outputs_list.append(outputs.detach().cpu().numpy())
            targets_list.append(targets.detach().cpu().numpy())

            batch_time_m.update(time.time() - end_time)

            if (idx+1) % log_eval_iter == 0:
                _logger.info('{name} [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                            'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                            'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            (idx+1), 
                            len(loader),
                            name       = name,
                            loss       = losses_m, 
                            batch_time = batch_time_m,
                            rate       = inputs.size(0) / batch_time_m.val,
                            rate_avg   = inputs.size(0) / batch_time_m.avg,
                            data_time  = data_time_m))
                    
            end_time = time.time()

        total_outputs[(model_key, area_grp_id)] = np.concatenate(outputs_list, axis=0)
        total_targets[(model_key, area_grp_id)] = np.concatenate(targets_list, axis=0)
    
    results = OrderedDict([('loss', losses_m.avg)])

    if return_avg:
        # 동일 지역(두 번째 요소의 접두어)을 기준으로 그룹화하여 평균 계산
        region_outputs = {}
        region_targets = {}
        for (model_key, area_grp_id), outputs in total_outputs.items():
            # area_grp_id에 '_'가 있으면 접두어 추출, 없으면 그대로 사용
            region = area_grp_id.split('_')[0]
            region_outputs.setdefault(region, []).append(outputs)
            region_targets.setdefault(region, []).append(total_targets[(model_key, area_grp_id)])
        for region, out_list in region_outputs.items():
            region_avg_output = np.concatenate(out_list, axis=0)
            region_avg_target = np.concatenate(region_targets[region], axis=0)
            region_metrics = cal_metric(pred=region_avg_output, true=region_avg_target, region=region)
            for k, v in region_metrics.items():
                results[f'avg_{region}_{k}'] = v
            _logger.info(f"📊 Average {name} Metrics for region {region}: {region_metrics}")

    if return_individual_outputs:
        individual_metrics = {}
        for (model_key, area_grp_id), outputs in total_outputs.items():
            region = area_grp_id.split('_')[0]
            metrics = cal_metric(pred=outputs, true=total_targets[(model_key, area_grp_id)], region=region)
            for k, v in metrics.items():
                results[f'{model_key}_{area_grp_id}_{k}'] = v
            individual_metrics[(model_key, area_grp_id)] = metrics
        _logger.info(f"📊 Individual {name} Metrics: {individual_metrics}")
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and return_output:
        os.makedirs(savedir, exist_ok=True)
        np.save(os.path.join(savedir, f'{name}_total_outputs.npy'), total_outputs)
        np.save(os.path.join(savedir, f'{name}_total_targets.npy'), total_targets)
    
    return results