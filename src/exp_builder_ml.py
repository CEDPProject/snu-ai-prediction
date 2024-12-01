import numpy as np
from utils.utils import cal_metric
from collections import defaultdict

def merge_predictions(predictions, group_key, eval_merge_grp, category_name=None):
    """
    Merge predictions by averaging based on a single group_key or its prefix if needed.
    """
    if not eval_merge_grp:
        return predictions

    group = group_key
    if category_name == "wind":
        group = group_key.split("_")[0]

    merged_results = {group: predictions}
    return {group: np.mean(merged_results[group], axis=0)}

def training_ml(model_class, trn_loader, batch_size=1024, eval_merge_grp=False, category_name=None):
    """
    Train machine learning models for each dataset combination in the training loader,
    with batch processing for memory efficiency during training.
    """
    results = {}
    model_dict = {}

    for idx, (model_key, area_dict) in enumerate(trn_loader.items(), start=1):
        for area_grp_id, data_dict in area_dict.items():
            print(f"Processing ({model_key}, {area_grp_id}) ({idx}/{len(trn_loader)})")

            # Extract train_x and train_y directly from data_dict
            train_x = data_dict['train_x'].drop(columns=["AREA_GRP_ID"], errors="ignore").values
            train_y = data_dict['train_y']["QGEN"].values

            # Initialize model
            model_instance = model_class
            model_dict[(model_key, area_grp_id)] = model_instance

            print(f"Fitting model for ({model_key}, {area_grp_id})")

            # Batch training
            num_samples = train_x.shape[0]
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_x = train_x[start_idx:end_idx]
                batch_y = train_y[start_idx:end_idx]

                # Fit model on the current batch
                model_instance.fit(batch_x, batch_y)

            # Prediction and metric calculation (using all training data for evaluation)
            predictions = model_instance.model.predict(train_x)

            if eval_merge_grp:
                print(f"Merging predictions for {area_grp_id}")
                merged_predictions = merge_predictions(predictions, area_grp_id, eval_merge_grp, category_name)
                metrics = cal_metric(pred=list(merged_predictions.values()), true=train_y)
            else:
                metrics = cal_metric(pred=predictions, true=train_y)

            print(f"Metrics for ({model_key}, {area_grp_id}): {metrics}")
            results[(model_key, area_grp_id)] = metrics

    return model_dict

def test_ml(model_dict, tst_loader, eval_merge_grp=False, category_name=None):
    results = {}
    grouped_metrics = defaultdict(list)  # Store metrics for each group

    for idx, (model_key, area_dict) in enumerate(tst_loader.items(), start=1):
        for area_grp_id, data_dict in area_dict.items():
            print(f"Testing ({model_key}, {area_grp_id}) ({idx}/{len(tst_loader)})")

            # Extract test_x and test_y directly from data_dict
            test_x = data_dict['test_x'].drop(columns=["AREA_GRP_ID"], errors="ignore").values
            test_y = data_dict['test_y']["QGEN"].values

            # Retrieve the trained model
            model_instance = model_dict.get((model_key, area_grp_id))
            if model_instance is None:
                raise ValueError(f"No trained model found for ({model_key}, {area_grp_id})")

            # Prediction and metric calculation
            predictions = model_instance.model.predict(test_x)

            # Grouping logic if eval_merge_grp is True
            if eval_merge_grp:
                group_key = area_grp_id.split('_')[0]  # Use prefix (e.g., "8711" from "8711_001")
                grouped_metrics[group_key].append((predictions, test_y))
            else:
                metrics = cal_metric(pred=predictions, true=test_y)
                print(f"Metrics for ({model_key}, {area_grp_id}): {metrics}")
                results[(model_key, area_grp_id)] = metrics

    if eval_merge_grp:
        print("Calculating average metrics for grouped areas...")
        for group_key, pred_true_pairs in grouped_metrics.items():
            all_predictions = []
            all_truths = []

            # Concatenate all predictions and ground truths for the group
            for predictions, truths in pred_true_pairs:
                all_predictions.extend(predictions)
                all_truths.extend(truths)

            # Convert to NumPy arrays
            all_predictions = np.array(all_predictions)
            all_truths = np.array(all_truths)

            # Calculate metrics for the merged group
            merged_metrics = cal_metric(pred=all_predictions, true=all_truths)
            print(f"Metrics for group {group_key}: {merged_metrics}")
            results[group_key] = merged_metrics

    return results