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

def training_dl(
    model, trainloader, validloader, criterion, optimizer, accelerator: Accelerator, 
    epochs: int, eval_epochs: int, log_epochs: int, log_eval_iter: int, wandb_iter: int,
    use_wandb: bool, ckp_metric: str, savedir: str, model_name: str, 
    pre_training:int, early_stopping_count: int, return_output: bool):

    early_stopping = EarlyStopping(patience=early_stopping_count)
    best_score = np.inf

    for (model_key, area_grp_id), loader in trainloader.items():
        _logger.info(f"🔹 Training on {model_key} - {area_grp_id}")
        
        batch_time_m = AverageMeter()
        data_time_m = AverageMeter()
        losses_m = AverageMeter()

        model.train()
        optimizer.zero_grad()
        end_time = time.time()

        early_stopping = EarlyStopping(patience=early_stopping_count)
        best_score = np.inf
        wandb_iteration = 0

        for epoch in range(epochs):
            for idx, item in enumerate(loader):
                data_time_m.update(time.time() - end_time)
                
                input = item['inputs'].float()
                target = item['targets'].float()
                
                outputs = model(input)
                loss = criterion(outputs, target)
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                batch_time_m.update(time.time() - end_time)
                wandb_iteration += 1

                if use_wandb and (wandb_iteration+1) % wandb_iter:
                    train_results = OrderedDict([
                        ('lr', optimizer.param_groups[0]['lr']),
                        ('train_loss', losses_m.avg)
                    ])
                    wandb.log(train_results, step=idx+1)
            
            if (epoch+1) % log_epochs == 0:
                _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                            'LR: {lr:.3e} '
                            'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                            'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            (idx+1), 
                            len(loader), 
                            loss       = losses_m, 
                            lr         = optimizer.param_groups[0]['lr'],
                            batch_time = batch_time_m,
                            rate       = input.size(0) / batch_time_m.val if batch_time_m.val > 0 else 0,
                            rate_avg   = input.size(0) / batch_time_m.avg if batch_time_m.avg > 0 else 0,
                            data_time  = data_time_m))

            if (epoch + 1) % eval_epochs == 0:
                if (model_key, area_grp_id) in validloader:
                    eval_metrics = test_dl(
                        model                     =   model, 
                        dataloader                =   { (model_key, area_grp_id): validloader[(model_key, area_grp_id)] },
                        criterion                 =   criterion,
                        name                      =   'VALID',
                        log_eval_iter             =   log_eval_iter,
                        pre_training              =   pre_training,
                        savedir                   =   savedir,
                        model_name                =   model_name,
                        accelerator               =   accelerator,
                        return_output             =   False,
                        return_avg                =   True,
                        return_individual_outputs =   False
                    )

                    model.train()
                    
                    eval_results = dict([(f'eval_{k}', v) for k, v in eval_metrics.items()])

                    if use_wandb:
                        wandb.log(eval_results, step=idx+1)

                    if best_score > eval_metrics[ckp_metric]:
                        state = {'best_epoch': epoch, 'best_step': idx+1, f'best_{ckp_metric}': eval_metrics[ckp_metric]}
                        
                        print('Save best model complete, epoch: {0:}: Best metric has changed from {1:.5f} \
                            to {2:.5f}'.format(epoch, best_score, eval_metrics[ckp_metric]))
            
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            state.update(eval_results)
                            json.dump(state, open(os.path.join(savedir, f'best_results.json'),'w'), indent='\t', cls=Float32Encoder)
                            torch.save(model.state_dict(), os.path.join(savedir, f'best_model_{model_key}_{area_grp_id}.pt'))
                            _logger.info(f"✅ Saved best model for {model_key} - {area_grp_id}")
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
            savedir: str, model_name: str, name: str = 'TEST', return_output: bool = False, return_avg: bool = True, 
            return_individual_outputs: bool = False):
    
    
    _logger.info(f'\n[🔍 Start {name} Evaluation]')

    # avereage meter
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    
    total_targets = {}
    total_outputs = {}

    end_time = time.time()
    wandb_iteration = 0

    model.eval()
    with torch.no_grad():
        for (model_key, area_grp_id), loader in dataloader.items():
            _logger.info(f"🧪 Testing on {model_key} - {area_grp_id}")
            outputs_list, targets_list = [], []
            for idx, item in enumerate(loader):
                data_time_m.update(time.time() - end_time)

                # input setting
                input = item['inputs'].float()
                target = item['targets'].float()
                
                # cal output 
                outputs = model(input)
                
                loss = criterion(outputs, target)
                loss = accelerator.gather(loss).mean()

                # Collect loss statistics
                outputs, target = accelerator.gather_for_metrics((outputs.contiguous(), target.contiguous()))
                losses_m.update(loss.item(), n=target.size(0))
                
                outputs_list.append(outputs.detach().cpu().numpy())
                targets_list.append(target.detach().cpu().numpy())

                # batch time        
                batch_time_m.update(time.time() - end_time)

                if (idx+1) % log_eval_iter == 0:
                    _logger.info('{name} [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                                'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                                (idx+1), 
                                len(dataloader),
                                name       = name,
                                loss       = losses_m, 
                                batch_time = batch_time_m,
                                rate       = input.size(0) / batch_time_m.val,
                                rate_avg   = input.size(0) / batch_time_m.avg,
                                data_time  = data_time_m))
                    
                end_time = time.time()

            total_outputs[(model_key, area_grp_id)] = np.concatenate(outputs_list, axis=0)
            total_targets[(model_key, area_grp_id)] = np.concatenate(targets_list, axis=0)
    
    results = OrderedDict([('loss',losses_m.avg)])
            
    # Compute metrics
    if return_avg:
        avg_output = np.mean([v for v in total_outputs.values()], axis=0)
        avg_target = np.mean([v for v in total_targets.values()], axis=0)
        avg_metrics = cal_metric(pred=avg_output, true=avg_target)
        results.update([(f'avg_{k}', v) for k, v in avg_metrics.items()])
        _logger.info(f"📊 Average {name} Metrics: {avg_metrics}")

    if return_individual_outputs:
        individual_metrics = {}
        for (model_key, area_grp_id), outputs in total_outputs.items():
            metrics = cal_metric(pred=outputs, true=total_targets[(model_key, area_grp_id)])
            results.update([(f'{model_key}_{area_grp_id}_{k}', v) for k, v in metrics.items()])
            individual_metrics[(model_key, area_grp_id)] = metrics
        _logger.info(f"📊 Individual {name} Metrics: {individual_metrics}")
    
    # Save results if required
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and return_output:
        os.makedirs(savedir, exist_ok=True)
        np.save(os.path.join(savedir, f'{name}_total_outputs.npy'), total_outputs)
        np.save(os.path.join(savedir, f'{name}_total_targets.npy'), total_targets)
    
    return results