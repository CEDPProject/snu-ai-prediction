from pycaret.regression import *
import pandas as pd
import numpy as np
from collections import defaultdict
from utils.utils import cal_metric
from sklearn.metrics import mean_squared_error, mean_absolute_error


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

def training_ml(model_class, trn_loader, eval_merge_grp=False, category_name=None):
    """
    Train machine learning models for each dataset combination in the training loader.
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
            model_instance.fit(train_x, train_y)

            # Prediction and metric calculation
            predictions = model_instance.model.predict(train_x)

            if eval_merge_grp:
                print(f"Merging predictions for {area_grp_id}")
                merged_predictions = merge_predictions(predictions, area_grp_id, eval_merge_grp, category_name)
                metrics = cal_metric(pred=list(merged_predictions.values()), true=train_y)
            else:
                metrics = cal_metric(pred=predictions, true=train_y)

            # print(f"Metrics for ({model_key}, {area_grp_id}): {metrics}")

    return model_dict

def test_ml(model_dict, tst_loader, eval_merge_grp=False, category_name=None):
    """
    Test machine learning models for each dataset combination in the test loader.
    Supports grouped evaluation and prediction merging.

    Args:
        model_dict: Dictionary containing trained models for each group.
        tst_loader: Test data loader containing test_x and test_y.
        eval_merge_grp: Whether to merge predictions for grouped evaluation.
        category_name: Category name for specific merging logic.

    Returns:
        results: Dictionary of calculated metrics for each group or merged group.
    """
    results = {}
    grouped_predictions = defaultdict(list)  # Store predictions and truths for grouped evaluation

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

            # Prediction
            predictions = model_instance.model.predict(test_x)

            if eval_merge_grp:
                # Group key handling
                group_key = area_grp_id.split('_')[0]  # Example: "8711_001" → "8711"
                grouped_predictions[group_key].append((predictions, test_y))
            else:
                # Calculate metrics without merging
                metrics = cal_metric(pred=predictions, true=test_y)
                print(f"Metrics for ({model_key}, {area_grp_id}): {metrics}")
                results[(model_key, area_grp_id)] = metrics

    # Process grouped metrics if grouping is enabled
    if eval_merge_grp:
        print("Calculating grouped metrics with merged predictions...")
        for group_key, pred_true_pairs in grouped_predictions.items():
            all_predictions = []
            all_truths = []

            # Merge predictions and ground truths for the group
            for predictions, truths in pred_true_pairs:
                all_predictions.extend(predictions)
                all_truths.extend(truths)

            # Use merge_predictions to average results (if needed)
            merged_predictions = merge_predictions(np.array(all_predictions), group_key, eval_merge_grp, category_name)
            all_truths = np.array(all_truths)

            # Calculate metrics for the merged group
            metrics = cal_metric(pred=merged_predictions[group_key], true=all_truths)
            print(f"Metrics for group {group_key}: {metrics}")
            results[group_key] = metrics

    return results

def training_ml_pycaret(
    trn_loader,
    selected_models=None,
    fold=5,
    save_dir='./saved_model/',
    enable_tuning=True,  # 하이퍼파라미터 튜닝 옵션 추가
    tuning_iter=10  # 튜닝 반복 횟수 설정
):
    """
    Train specified machine learning models using PyCaret for each dataset combination in the training loader.
    Optionally perform hyperparameter tuning for each model and save all hyperparameter configurations and results.
    """
    
    all_model_results = []  # List to store all model performance
    all_trained_models = {}  # Dictionary to store all trained models

    for idx, (model_key, area_dict) in enumerate(trn_loader.items(), start=1):
        for area_grp_id, data_dict in area_dict.items():
            print(f"Processing ({model_key}, {area_grp_id}) ({idx}/{len(trn_loader)})")

            # Combine train_x and train_y into a single DataFrame
            train_x = data_dict['train_x']
            train_y = data_dict['train_y']['QGEN']
            train_data = pd.concat([train_x, train_y], axis=1)

            # Set up PyCaret for regression
            print(f"Setting up PyCaret for ({model_key}, {area_grp_id})...")
            reg_setup = setup(
                data=train_data,
                target='QGEN',
                fold=fold,
                verbose=False,
                html=False
            )

            # Get available models
            available_models = models()
            if selected_models is None:
                model_list = available_models.index.tolist()  # Use all available models
            else:
                model_list = [m for m in selected_models if m in available_models.index]  # Filter by selected models

            # Train specified models, optionally tune their hyperparameters, and store their performance
            for model_name in model_list:
                model_desc = available_models.loc[model_name]
                print(f"Training model: {model_name} ({model_desc['Name']})")
                
                # Train the model
                trained_model = create_model(model_name, verbose=False)
                tuned_params = None  # Default for cases without tuning
                
                if enable_tuning:
                    # Perform hyperparameter tuning
                    print(f"Tuning model: {model_name}...")
                    tuned_model = tune_model(trained_model, verbose=False, n_iter=tuning_iter)
                    # Extract tuned hyperparameters
                    tuned_params = tuned_model.get_params()
                    final_model = tuned_model
                else:
                    final_model = trained_model

                # Get the performance metrics
                model_results = pull().to_dict(orient="records")[0]  # Convert PyCaret pull() result to dict

                # Save individual tuned parameters and performance
                model_results['model_key'] = model_key
                model_results['area_grp_id'] = area_grp_id
                model_results['model_name'] = model_name
                model_results['tuned_params'] = str(tuned_params)  # Convert tuned_params to string for saving
                all_model_results.append(model_results)  # Append results

                # Store the final model
                all_trained_models[(model_key, area_grp_id, model_name, str(tuned_params))] = final_model

    # Combine all model results into a single DataFrame
    results_df = pd.DataFrame(all_model_results)
    results_df.to_csv(f'{save_dir}/training_results_with_tuning.csv', index=False)

    return all_trained_models


def test_ml_pycaret(
    all_trained_models,
    tst_loader,
    target_column="QGEN",
    eval_merge_grp=False,
    category_name=None,
    save_dir="./saved_model/",
):
    """
    Test trained PyCaret models on given test data and calculate metrics.
    Record predictions and save the results, including MSE, MAE, MAPE, and tuned hyperparameters for each model and group.
    """
    metrics_results = []  # List to store metrics for all models
    grouped_metrics = defaultdict(lambda: defaultdict(list))  # Store predictions per group and model

    for idx, (model_key, area_dict) in enumerate(tst_loader.items(), start=1):
        for area_grp_id, data_dict in area_dict.items():
            print(f"Testing ({model_key}, {area_grp_id}) ({idx}/{len(tst_loader)})")

            # Combine test_x and test_y into a single DataFrame
            test_x = data_dict["test_x"]
            test_y = data_dict["test_y"][target_column]
            test_data = pd.concat([test_x, test_y], axis=1)

            # Test all models trained for this group
            for (m_key, a_grp_id, model_name, tuned_params), trained_model in all_trained_models.items():
                if (m_key, a_grp_id) == (model_key, area_grp_id):
                    print(f"Testing model: {model_name} for ({model_key}, {area_grp_id})")
                    predictions = predict_model(trained_model, data=test_data, verbose=False)

                    # Extract predicted values
                    pred_values = predictions["prediction_label"].values
                    true_values = predictions[target_column].values

                    # Calculate metrics directly for this model
                    metrics = cal_metric(pred=pred_values, true=true_values)
                    metrics_results.append({
                        'model_key': model_key,
                        'area_grp_id': area_grp_id,
                        'model_name': model_name,
                        'MSE': metrics['MSE'],
                        'MAE': metrics['MAE'],
                        'MAPE': metrics['MAPE'],
                        'tuned_params': tuned_params  # Save tuned hyperparameters
                    })

    # Save metrics
    metrics_results_df = pd.DataFrame(metrics_results)
    metrics_results_df.to_csv(f'{save_dir}/test_results_with_tuning.csv', index=False)

    return metrics_results_df