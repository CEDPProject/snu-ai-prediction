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

    with torch.no_grad():
        for (model_key, area_grp_id), loader in dataloader.items():
            _logger.info(f"🧪 Testing on {model_key} - {area_grp_id}")
            outputs_list, targets_list = [], []
            if isinstance(model, dict):
                current_model = model[(model_key, area_grp_id)]
            else:
                current_model = model
            current_model.eval()
            
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
        import pandas as pd
        from collections import defaultdict

        # === 지역별 model_key별 누적 구조 생성 ===
        region_preds_by_model = defaultdict(lambda: defaultdict(list))
        region_trues_by_model = defaultdict(lambda: defaultdict(list))

        for (model_key, area_grp_id), outputs in total_outputs.items():
            region = area_grp_id.split('_')[0]
            region_preds_by_model[model_key][region].append(outputs)
            region_trues_by_model[model_key][region].append(total_targets[(model_key, area_grp_id)])

        region_model_metrics_list = []
        for model_key in region_preds_by_model:
            for region in region_preds_by_model[model_key]:
                # 각 지역-모델 조합에서 시점별 합산
                summed_pred = np.sum(region_preds_by_model[model_key][region], axis=0)
                summed_true = np.sum(region_trues_by_model[model_key][region], axis=0)

                # 지표 계산 (정격출력 10% 이상인 시점만 사용)
                region_metrics = cal_metric(pred=summed_pred, true=summed_true, region=region)

                # 결과 저장
                for k, v in region_metrics.items():
                    results[f'{model_key}_{region}_{k}'] = v

                region_model_metrics_list.append({
                    'model_key': model_key,
                    'region': region,
                    **region_metrics
                })

                _logger.info(f"📊 {name} Metrics for model {model_key} - region {region}: {region_metrics}")

        # === DataFrame 저장 ===
        region_model_df = pd.DataFrame(region_model_metrics_list)
        region_model_df.to_csv(os.path.join(savedir, f"{name}_region_model_metrics.csv"), index=False)

        # === 모델별 지역 평균 지표 계산 ===
        metric_cols = ['MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE', 'RSE', 'NMAE']
        region_avg_df = (
            region_model_df
            .groupby('region', as_index=False)[metric_cols]
            .mean()
            .rename(columns=lambda x: f'avg_{x}' if x != 'region' else x)
        )
        region_avg_df.to_csv(os.path.join(savedir, f"{name}_region_avg_metrics.csv"), index=False)

        # 결과 딕셔너리에 포함
        for _, row in region_avg_df.iterrows():
            for k in ['avg_MAE', 'avg_MSE', 'avg_RMSE', 'avg_MAPE', 'avg_MSPE', 'avg_RSE', 'avg_NMAE']:
                results[f'{row["region"]}_{k}'] = row[k]

    # (기존 개별 출력 로직은 유지 가능)
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