import numpy as np
from utils.utils import cal_metric, prepare_data

from collections import defaultdict
import numpy as np

def merge_predictions(predictions, group_keys, eval_merge_grp, category_name=None):
    """
    Merge predictions by averaging based on area_grp_id or its prefix if needed.
    """
    if not eval_merge_grp:
        return predictions

    merged_results = defaultdict(list)

    for pred, group in zip(predictions, group_keys):
        if category_name == "wind":
            group = group.split("_")[0]  # Use prefix (e.g., 5375 from 5375_001)
        merged_results[group].append(pred)

    # Average predictions for each group
    return {group: np.mean(vals, axis=0) for group, vals in merged_results.items()}


def training_ml(model_class, trn_loader, eval_merge_grp=False, category_name=None):
    results = {}
    total_combinations = len(trn_loader)
    model_dict = {}

    for idx, ((model_key, area_grp_id), current_loader) in enumerate(trn_loader.items(), start=1):
        print(f"Processing combination: ({model_key}, {area_grp_id}) ({idx}/{total_combinations})")
        
        # Load or process data
        all_x, all_y = prepare_data(current_loader)
        
        print(f"Initializing model for ({model_key}, {area_grp_id})")
        # Initialize and save the model
        model_instance = model_class
        model_dict[(model_key, area_grp_id)] = model_instance
        
        print(f"Fitting model for ({model_key}, {area_grp_id})")
        model_instance.model.fit(all_x, all_y)

        print(f"Starting prediction")
        # Collect predictions
        all_predictions = []
        group_keys = []
        for batch in current_loader:
            x = batch['inputs']
            x_np = x.detach().cpu().numpy()
            predictions = model_instance.predict(x_np)
            all_predictions.append(predictions)
            group_keys.append(area_grp_id)  # Store the group key for merging
        
        all_predictions = np.concatenate(all_predictions)
        all_y = np.array(all_y)

        if eval_merge_grp:
            print(f"Merging predictions for area_grp_id={area_grp_id}")
            merged_predictions = merge_predictions(all_predictions, group_keys, eval_merge_grp, category_name)
            # Calculate metrics for merged results
            metrics = cal_metric(pred=list(merged_predictions.values()), true=all_y)
        else:
            metrics = cal_metric(pred=all_predictions, true=all_y)

        print(f"Completed processing for ({model_key}, {area_grp_id})")
        print(f"metrics: {metrics}")
        results[(model_key, area_grp_id)] = metrics

    return model_dict, results


def test_ml(model_dict, tst_loader, eval_merge_grp=False, category_name=None):
    results = {}
    total_combinations = len(tst_loader)

    for idx, ((model_key, area_grp_id), current_loader) in enumerate(tst_loader.items(), start=1):
        print(f"Processing combination: ({model_key}, {area_grp_id}) ({idx}/{total_combinations})")
        
        # Load or process data
        all_x, all_y = prepare_data(current_loader)
        
        if (model_key, area_grp_id) not in model_dict:
            raise ValueError(f"No trained model found for ({model_key}, {area_grp_id})")
        
        model_instance = model_dict[(model_key, area_grp_id)]
        
        print(f"Starting prediction for ({model_key}, {area_grp_id})")
        all_predictions = []
        group_keys = []
        for batch in current_loader:
            x = batch['inputs']
            x_np = x.detach().cpu().numpy()
            predictions = model_instance.model.predict(x_np)
            all_predictions.append(predictions)
            group_keys.append(area_grp_id)  # Store group key for merging
        
        all_predictions = np.concatenate(all_predictions)
        all_y = np.array(all_y)

        if eval_merge_grp:
            print(f"Merging predictions for area_grp_id={area_grp_id}")
            merged_predictions = merge_predictions(all_predictions, group_keys, eval_merge_grp, category_name)
            metrics = cal_metric(pred=list(merged_predictions.values()), true=all_y)
        else:
            metrics = cal_metric(pred=all_predictions, true=all_y)

        print(f"Metrics for ({model_key}, {area_grp_id}): {metrics}")
        results[(model_key, area_grp_id)] = metrics

    return results