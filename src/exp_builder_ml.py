import torch
import numpy as np
from utils.utils import cal_metric

def training_ml(model, trn_loader):
    results = {}
    total_combinations = len(trn_loader)
    
    for idx, ((model_key, area_grp_id), current_loader) in enumerate(trn_loader.items(), start=1):
        print(f"Processing combination: ({model_key}, {area_grp_id}) ({idx}/{total_combinations})")
        
        # Collect predictions and targets
        all_x, all_y = [], []
        total_batches = len(current_loader)
        
        for batch_idx, batch in enumerate(current_loader, start=1):
            # Progress update
            if batch_idx % 100 == 0 or batch_idx == total_batches:
                print(f"Processing... {batch_idx}/{total_batches} ({(batch_idx / total_batches) * 100:.2f}%)")
            
            x, y = batch['inputs'], batch['targets']
            all_x.append(x.detach().cpu().numpy())
            all_y.append(y.detach().cpu().numpy())
        
        print(f"set data for ML")
        all_x = np.vstack(all_x)
        all_y = np.hstack(all_y)
        
        print(f"Fitting model")
        # Train the model
        model.ensemble.fit(all_x, all_y)
        
        print(f"Starting prediction")
        # Collect predictions
        all_predictions = []
        for batch in current_loader:
            x = batch['inputs']
            x_np = x.detach().cpu().numpy()
            predictions = model.ensemble.predict(x_np)
            all_predictions.append(predictions)
        
        all_predictions = np.concatenate(all_predictions)
        
        # Calculate metrics
        metrics = cal_metric(pred=all_predictions, true=all_y)

        print(f"Completed processing for ({model_key}, {area_grp_id})")
        print(f"metrics: {metrics}")

def test_ml(model, tst_loader):
    results = {}
    total_combinations = len(tst_loader)
    
    for idx, ((model_key, area_grp_id), current_loader) in enumerate(tst_loader.items(), start=1):
        print(f"Processing combination: ({model_key}, {area_grp_id}) ({idx}/{total_combinations})")

        # Collect predictions and targets
        all_predictions = []
        all_y = []
        total_batches = len(current_loader)
        for batch_idx, batch in enumerate(current_loader, start=1):
            # Progress update
            if batch_idx % 100 == 0 or batch_idx == total_batches:
                print(f"Processing... {batch_idx}/{total_batches} ({(batch_idx / total_batches) * 100:.2f}%)")
            x, y = batch['inputs'], batch['targets']
            x_np = x.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            predictions = model.model.predict(x_np)
            all_predictions.append(predictions)
            all_y.append(y_np)
        
        all_predictions = np.concatenate(all_predictions)
        all_y = np.concatenate(all_y)
        
        # Calculate metrics
        metrics = cal_metric(pred=all_predictions, true=all_y)
        
        print(f"Metrics for ({model_key}, {area_grp_id}): {metrics}")

        # Save results
        results[(model_key, area_grp_id)] = {
            'predictions': all_predictions,
            'metrics': metrics
        }

    return metrics