from pycaret.regression import *
import pandas as pd
import numpy as np
from collections import defaultdict
from utils.utils import cal_metric
from sklearn.metrics import mean_squared_error, mean_absolute_error
from omegaconf import OmegaConf
from sklearn.model_selection import ParameterGrid
import os

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
    grid_model,
    selected_models=None,
    k_fold_num=5,
    save_dir='./saved_model/',
    enable_tuning=True,  # 하이퍼파라미터 튜닝 옵션 추가
    tuning_iter=10,  # 튜닝 반복 횟수 설정
    optimize_metric='MSE',
    search_library='optuna',
    search_algorithm='random',  # 기본값은 'random'
    early_stopping=False,
    n_jobs=8,
    category_name='solar'
):
    """
    Train specified machine learning models using PyCaret for each dataset combination in the training loader.
    Perform hyperparameter tuning (Grid Search or Random Search) and save all configurations and results.
    """
    
    all_model_results = []  # Store all model performance
    all_trained_models = {}  # Store all trained models
    all_grid_results = []
    grid_model = OmegaConf.to_container(grid_model, resolve=True)
    feature_names_dict = {}  # Dictionary to store feature names for each (model_key, area_grp_id)

    for idx, (model_key, area_dict) in enumerate(trn_loader.items(), start=1):
        for area_grp_id, data_dict in area_dict.items():
            print(f"Processing ({model_key}, {area_grp_id}) ({idx}/{len(trn_loader)})")

            # Combine train_x and train_y into a single DataFrame
            train_x = data_dict['train_x']
            train_x = train_x.drop(columns=["AREA_GRP_ID"], errors="ignore")
            train_y = data_dict['train_y']['QGEN']
            train_data = pd.concat([train_x, train_y], axis=1)

            # Save feature names for this group
            feature_names_dict[(model_key, area_grp_id)] = train_x.columns.tolist()

            # Set up PyCaret for regression
            print(f"Setting up PyCaret for ({model_key}, {area_grp_id})...")
            reg_setup = setup(
                data=train_data,
                target='QGEN',
                fold=k_fold_num,
                verbose=False,
                html=False,
                n_jobs=n_jobs,
                session_id=42 
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
                params = None  # Default for cases without tuning
                
                if enable_tuning:
                    if search_algorithm == 'grid':  # Grid Search
                        print(f"Performing Grid Search for {model_name}...")
                        if model_name not in grid_model:
                            raise ValueError(f"No grid found for model: {model_name}")
                        custom_grid = grid_model[model_name]

                        # Ensure all values in custom_grid are lists
                        custom_grid = {key: value if isinstance(value, list) else [value] for key, value in custom_grid.items()}

                        # Generate all parameter combinations
                        from sklearn.model_selection import ParameterGrid
                        param_grid = list(ParameterGrid(custom_grid))

                        # Track results for each parameter combination
                        grid_results = []
                        best_model = None
                        best_score = float('inf') if optimize_metric in ['MSE', 'RMSE', 'MAE'] else float('-inf')
                        best_params = None

                        for params in param_grid:
                            print(f"Testing parameters: {params}")
                            params = {key: [value] if not isinstance(value, list) else value for key, value in params.items()}
                            tuned_model = tune_model(
                                trained_model,
                                custom_grid=params,
                                verbose=False,
                                optimize=optimize_metric
                            )

                            # Get performance metrics
                            model_results = pull().to_dict(orient="records")[0]
                            model_results['params'] = params
                            model_results['model_key'] = model_key
                            model_results['area_grp_id'] = area_grp_id
                            model_results['model_name'] = model_name
                            all_grid_results.append(model_results)

                            # Check for the best model
                            current_score = model_results[optimize_metric]
                            if (optimize_metric in ['MSE', 'RMSE', 'MAE'] and current_score < best_score) or \
                               (optimize_metric not in ['MSE', 'RMSE', 'MAE'] and current_score > best_score):
                                best_score = current_score
                                best_model = tuned_model
                                best_params = params

                    elif search_algorithm == 'random':  # Random Search
                        print(f"Performing Random Search for {model_name}...")
                        tuned_model = tune_model(
                            trained_model,
                            custom_grid=grid_model.get(model_name, {}),
                            verbose=False,
                            n_iter=tuning_iter,
                            optimize=optimize_metric,
                            search_library=search_library,
                            search_algorithm=search_algorithm,
                            early_stopping=early_stopping
                        )
                        best_model = tuned_model
                        best_params = tuned_model.get_params()

                    final_model = best_model
                else:
                    params = trained_model.get_params()
                    final_model = trained_model

                # Get the performance metrics
                model_results = pull().to_dict(orient="records")[0]  # Convert PyCaret pull() result to dict
                model_results['model_key'] = model_key
                model_results['area_grp_id'] = area_grp_id
                model_results['model_name'] = model_name
                if enable_tuning:
                    params = str(best_params)
                    model_results['params'] = params
                else:
                    model_results['params'] = str('default')
                all_model_results.append(model_results)

                # Store the final trained model
                all_trained_models[(model_key, area_grp_id, model_name, params)] = final_model

    final_results_df = pd.DataFrame(all_grid_results)
    final_results_df["params"] = final_results_df["params"].astype(str)
    if category_name == 'wind':
        final_results_df['area_grp_id'] = final_results_df['area_grp_id'].apply(lambda x: x.split('_')[0])
    final_results_df = final_results_df[["area_grp_id", "model_name", "params", "MSE", "MAE", "MAPE", "RMSE", "R2", "RMSLE"]]
    aggregated_df = final_results_df.groupby(["params", "area_grp_id", "model_name"]).mean().reset_index()
    aggregated_df.to_csv(f'{save_dir}/grid_results.csv', index=False)

    # Combine all model results into a single DataFrame
    results_df = pd.DataFrame(all_model_results)
    results_df.to_csv(f'{save_dir}/training_results_with_tuning.csv', index=False)

    return all_trained_models, feature_names_dict

def test_ml_pycaret(
    all_trained_models,
    tst_loader,
    feature_names_dict,
    target_column="QGEN",
    save_dir="./saved_model/",
    eval_merge_grp = True,
    category_name = 'solar'
):
    """
    Test trained PyCaret models on given test data and calculate metrics.
    Record predictions and save the results, including MSE, MAE, MAPE, and tuned hyperparameters for each model and group.
    """
    metrics_results = []  # List to store metrics for all models

    for idx, (model_key, area_dict) in enumerate(tst_loader.items(), start=1):
        for area_grp_id, data_dict in area_dict.items():
            print(f"Testing ({model_key}, {area_grp_id}) ({idx}/{len(tst_loader)})")
            
            # Combine test_x and test_y into a single DataFrame
            test_x = data_dict["test_x"]
            test_x = test_x.drop(columns=["AREA_GRP_ID"], errors="ignore")
            test_y = data_dict["test_y"][target_column]
            
            # Align test features with training features
            expected_features = feature_names_dict.get((model_key, area_grp_id), [])
            for feature in expected_features:
                if feature not in test_x.columns:
                    test_x[feature] = 0  # Add missing feature with default value
                    
                # Remove extra features
            extra_features = set(test_x.columns) - set(expected_features)
            if extra_features:
                test_x = test_x.drop(columns=list(extra_features))

            # Reorder columns to match training features
            test_x = test_x[expected_features]

            test_data = pd.concat([test_x, test_y], axis=1)

            # Test all models trained for this group
            for (m_key, a_grp_id, model_name, params), trained_model in all_trained_models.items():
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
                        'params': params  # Save tuned hyperparameters
                    })

    # Save individual results
    metrics_results_df = pd.DataFrame(metrics_results)
    metrics_results_df.to_csv(f'{save_dir}/test_results_with_tuning.csv', index=False)

    # Check if category_name is 'wind' and extract prefix if needed
    if category_name == 'wind':
        metrics_results_df['area_grp_prefix'] = metrics_results_df['area_grp_id'].apply(lambda x: x.split('_')[0])
        group_column = 'area_grp_prefix'
    else:
        group_column = 'area_grp_id'

    # Group by 'model_name' and the appropriate group column
    averaged_metrics = (
        metrics_results_df.groupby(['model_name', group_column], as_index=False)
        .agg({
            'MSE': 'mean',
            'MAE': 'mean',
            'MAPE': 'mean',
            'model_key': lambda x: ', '.join(set(x)),
            'params': 'first'
        })
    )

    # Save averaged results
    averaged_metrics.to_csv(f'{save_dir}/test_results_with_tuning_averaged.csv', index=False)

    return averaged_metrics

