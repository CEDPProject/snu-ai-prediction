from pycaret.regression import *
import pandas as pd
import numpy as np
from collections import defaultdict
from utils.utils import cal_metric
from sklearn.metrics import mean_squared_error, mean_absolute_error
from omegaconf import OmegaConf
from sklearn.model_selection import ParameterGrid
import logging

logger = logging.getLogger('ML Models') 

def clamp_predictions(predictions):
    return np.maximum(predictions, 0)

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

def training_ml(model_class, trn_loader, clamping, eval_merge_grp=False, category_name=None, return_output=False):
    """
    Train machine learning models for each dataset combination in the training loader.
    """
    results = {}
    model_dict = {}

    for idx, (model_key, area_dict) in enumerate(trn_loader.items(), start=1):
        for area_grp_id, data_dict in area_dict.items():
            logger.info(f"Processing ({model_key}, {area_grp_id}) ({idx}/{len(trn_loader)})")

            # Extract train_x and train_y directly from data_dict
            train_x = data_dict['train_x'].drop(columns=["AREA_GRP_ID"], errors="ignore").values
            train_y = data_dict['train_y']["QGEN"].values

            # Initialize model
            model_instance = model_class
            model_dict[(model_key, area_grp_id)] = model_instance

            logger.info(f"Fitting model for ({model_key}, {area_grp_id})")
            model_instance.fit(train_x, train_y)

            # Prediction and metric calculation
            predictions = model_instance.model.predict(train_x)
            if clamping:
                predictions = clamp_predictions(predictions)

            if eval_merge_grp:
                logger.info(f"Merging predictions for {area_grp_id}")
                merged_predictions = merge_predictions(predictions, area_grp_id, eval_merge_grp, category_name)
                metrics = cal_metric(pred=list(merged_predictions.values()), true=train_y)
            else:
                metrics = cal_metric(pred=predictions, true=train_y)

            # print(f"Metrics for ({model_key}, {area_grp_id}): {metrics}")

    return model_dict

def test_ml(model_dict, tst_loader, clamping, eval_merge_grp=False, category_name=None, return_output=False, savedir=None):
    """
    Test machine learning models for each dataset combination in the test loader.
    Supports grouped evaluation and prediction merging.

    Args:
        model_dict: Dictionary containing trained models for each group.
        tst_loader: Test data loader containing test_x and test_y.
        eval_merge_grp: Whether to merge predictions for grouped evaluation.
        category_name: Category name for specific merging logic.
        return_output: If True, save predictions and targets as numpy files.
        savedir: Directory to save outputs (required if return_output=True).

    Returns:
        results: Dictionary of calculated metrics for each group or merged group.
    """
    results = {}
    grouped_predictions = defaultdict(list)  # Store predictions and truths for grouped evaluation
    
    # For saving outputs
    all_predictions = []
    all_targets = []

    for idx, (model_key, area_dict) in enumerate(tst_loader.items(), start=1):
        for area_grp_id, data_dict in area_dict.items():
            logger.info(f"Testing ({model_key}, {area_grp_id}) ({idx}/{len(tst_loader)})")

            # Extract test_x and test_y directly from data_dict
            test_x = data_dict['test_x'].drop(columns=["AREA_GRP_ID"], errors="ignore").values
            test_y = data_dict['test_y']["QGEN"].values

            # Retrieve the trained model
            model_instance = model_dict.get((model_key, area_grp_id))
            if model_instance is None:
                raise ValueError(f"No trained model found for ({model_key}, {area_grp_id})")

            # Prediction
            predictions = model_instance.model.predict(test_x)
            if clamping:
                predictions = clamp_predictions(predictions)

            # Collect for output saving
            if return_output:
                all_predictions.append(predictions)
                all_targets.append(test_y)

            if eval_merge_grp:
                # Group key handling
                group_key = area_grp_id.split('_')[0]  # Example: "8711_001" → "8711"
                grouped_predictions[group_key].append((predictions, test_y))
            else:
                region = area_grp_id.split('_')[0] 
                # Calculate metrics without merging
                metrics = cal_metric(pred=predictions, true=test_y, region=region)
                logger.info(f"Metrics for ({model_key}, {area_grp_id}): {metrics}")
                results[(model_key, area_grp_id)] = metrics

    # Process grouped metrics if grouping is enabled
    if eval_merge_grp:
        logger.info("Calculating grouped metrics with merged predictions...")
        for group_key, pred_true_pairs in grouped_predictions.items():
            all_predictions_group = []
            all_truths = []

            # Merge predictions and ground truths for the group
            for predictions, truths in pred_true_pairs:
                all_predictions_group.extend(predictions)
                all_truths.extend(truths)

            # Use merge_predictions to average results (if needed)
            merged_predictions = merge_predictions(np.array(all_predictions_group), group_key, eval_merge_grp, category_name)
            if clamping:
                merged_predictions = clamp_predictions(merged_predictions)
            all_truths = np.array(all_truths)
            region = group_key.split('_')[0] 
            # Calculate metrics for the merged group
            metrics = cal_metric(pred=merged_predictions[group_key], true=all_truths, region=region)
            logger.info(f"Metrics for group {group_key}: {metrics}")
            results[group_key] = metrics

    # Save outputs if requested
    if return_output:
        if savedir is None:
            raise ValueError("savedir must be provided when return_output=True")
        
        os.makedirs(savedir, exist_ok=True)
        total_outputs = np.concatenate(all_predictions, axis=0)
        total_targets = np.concatenate(all_targets, axis=0)
        
        np.save(os.path.join(savedir, 'TEST_total_outputs.npy'), total_outputs)
        np.save(os.path.join(savedir, 'TEST_total_targets.npy'), total_targets)
        logger.info(f"Saved predictions and targets to {savedir}")

    return results

def training_ml_pycaret(
    trn_loader,
    model_grids,
    ignore_features_list,
    selected_models=None,
    k_fold_num=5,
    save_dir='./saved_model/',
    enable_tuning=True,
    tuning_iter=10,
    optimize_metric='MSE',
    search_library='optuna',
    search_algorithm='random',
    early_stopping=False,
    clamping=False,
    n_jobs=8,
    category_name='solar',
    enable_ensemble=True,  # PyCaret 앙상블 활성화
    ensemble_types=['blend', 'stack', 'bagging'],  # 사용할 PyCaret 앙상블
    return_output=False
):
    """
    Enhanced training function with both individual models and PyCaret ensemble methods
    """
    
    all_model_results = []
    all_trained_models = {}
    all_grid_results = []
    all_ensemble_models = {}  # PyCaret 앙상블 모델 저장
    grid_model = OmegaConf.to_container(model_grids, resolve=True)
    feature_names_dict = {}

    for idx, (model_key, area_dict) in enumerate(trn_loader.items(), start=1):
        for area_grp_id, data_dict in area_dict.items():
            logger.info(f"Processing ({model_key}, {area_grp_id}) ({idx}/{len(trn_loader)})")
            
            # 데이터 준비
            train_x = data_dict['train_x']
            train_x = train_x.drop(columns=["AREA_GRP_ID"], errors="ignore")
            train_y = data_dict['train_y']['QGEN']
            train_data = pd.concat([train_x, train_y], axis=1)

            feature_names_dict[(model_key, area_grp_id)] = train_x.columns.tolist()

            # PyCaret 설정
            logger.info(f"Setting up PyCaret for ({model_key}, {area_grp_id})...")
            reg_setup = setup(
                data=train_data,
                target='QGEN',
                fold=k_fold_num,
                verbose=False,
                html=False,
                n_jobs=n_jobs,
                session_id=42,
                ignore_features=ignore_features_list
            )
            
            # 사용 가능한 모델 확인
            available_models = models()
            if selected_models is None:
                model_list = available_models.index.tolist()
            else:
                model_list = [m for m in selected_models if m in available_models.index]
            
            # 개별 모델 학습 (기존 코드 유지)
            base_models = []  # 앙상블을 위한 베이스 모델 저장
            model_names_for_ensemble = []
            
            for model_name in model_list:
                model_desc = available_models.loc[model_name]
                logger.info(f"Training model: {model_name} ({model_desc['Name']})")
                
                trained_model = create_model(model_name, verbose=False)
                params = None
                
                if enable_tuning:
                    if search_algorithm == 'grid':
                        logger.info(f"Performing Grid Search for {model_name}...")
                        if model_name not in grid_model:
                            raise ValueError(f"No grid found for model: {model_name}")
                        custom_grid = grid_model[model_name]
                        custom_grid = {key: value if isinstance(value, list) else [value] for key, value in custom_grid.items()}
                        
                        param_grid = list(ParameterGrid(custom_grid))
                        
                        for param_dict in param_grid:
                            for key, value in param_dict.items():
                                if value == 'None':
                                    param_dict[key] = None
                                elif value == 'True':
                                    param_dict[key] = True
                                elif value == 'False':
                                    param_dict[key] = False

                        grid_results = []
                        best_model = None
                        best_score = float('inf') if optimize_metric in ['MSE', 'RMSE', 'MAE'] else float('-inf')
                        best_params = None
                        
                        logger.info(f"Testing {len(param_grid)} parameter combinations...")
                        
                        for i, params in enumerate(param_grid, 1):
                            logger.info(f"[{i}/{len(param_grid)}] Testing parameters: {params}")
                            pycaret_params = {key: [value] for key, value in params.items()}
                            
                            try:
                                tuned_model = tune_model(
                                    trained_model,
                                    custom_grid=pycaret_params,
                                    verbose=False,
                                    optimize=optimize_metric
                                )

                                model_results = pull().to_dict(orient="records")[0]
                                model_results['params'] = params
                                model_results['model_key'] = model_key
                                model_results['area_grp_id'] = area_grp_id
                                model_results['model_name'] = model_name
                                all_grid_results.append(model_results)

                                current_mae = model_results['MAE']
                                if current_mae < best_score:
                                    best_score = current_mae
                                    best_model = tuned_model
                                    best_params = params
                                    logger.info(f"  ✅ New best MAE: {best_score:.4f}")
                                else:
                                    logger.info(f"  📊 MAE: {current_mae:.4f}")
                                    
                            except Exception as e:
                                logger.info(f"  ❌ Error with params {params}: {e}")
                                continue

                        logger.info(f"🎯 Best parameters for {model_name}: {best_params}")
                        logger.info(f"🎯 Best MAE: {best_score:.4f}")

                    elif search_algorithm == 'random':
                        logger.info(f"Performing Random Search for {model_name}...")
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

                # 성능 메트릭 저장
                model_results = pull().to_dict(orient="records")[0]
                model_results['model_key'] = model_key
                model_results['area_grp_id'] = area_grp_id
                model_results['model_name'] = model_name

                if enable_tuning:
                    params_dict = best_params if best_params is not None else {}
                    model_results['params'] = str(params_dict)
                else:
                    params_dict = trained_model.get_params()
                    model_results['params'] = str('default')

                all_model_results.append(model_results)

                model_id = len(all_trained_models)
                all_trained_models[(model_key, area_grp_id, model_name, model_id)] = final_model
                
                # 앙상블을 위한 베이스 모델 저장
                base_models.append(final_model)
                model_names_for_ensemble.append(model_name)
            
            # PyCaret 앙상블 모델 생성
            if enable_ensemble and len(base_models) >= 2:
                logger.info(f"🔧 Creating PyCaret ensemble models for ({model_key}, {area_grp_id})...")
                
                # Blend Models
                if 'blend' in ensemble_types:
                    try:
                        logger.info("Creating Blend ensemble...")
                        blended_model = blend_models(estimator_list=base_models, fold=k_fold_num, verbose=False)
                        all_ensemble_models[(model_key, area_grp_id, 'pycaret_blend')] = blended_model
                        
                        blend_results = pull().to_dict(orient="records")[0]
                        blend_results.update({
                            'model_key': model_key,
                            'area_grp_id': area_grp_id,
                            'model_name': 'pycaret_blend',
                            'base_models': ','.join(model_names_for_ensemble),
                            'params': 'blend_default'
                        })
                        all_model_results.append(blend_results)
                        logger.info(f"  ✅ Blend ensemble MAE: {blend_results['MAE']:.4f}")
                    except Exception as e:
                        logger.info(f"  ❌ Blend ensemble failed: {e}")
                
                # Stack Models
                if 'stack' in ensemble_types:
                    try:
                        logger.info("Creating Stack ensemble...")
                        meta_model = create_model('lr', verbose=False)
                        stacked_model = stack_models(
                            estimator_list=base_models,
                            meta_model=meta_model,
                            fold=k_fold_num,
                            verbose=False
                        )
                        all_ensemble_models[(model_key, area_grp_id, 'pycaret_stack')] = stacked_model
                        
                        stack_results = pull().to_dict(orient="records")[0]
                        stack_results.update({
                            'model_key': model_key,
                            'area_grp_id': area_grp_id,
                            'model_name': 'pycaret_stack',
                            'base_models': ','.join(model_names_for_ensemble),
                            'meta_model': 'lr',
                            'params': 'stack_with_lr'
                        })
                        all_model_results.append(stack_results)
                        logger.info(f"  ✅ Stack ensemble MAE: {stack_results['MAE']:.4f}")
                    except Exception as e:
                        logger.info(f"  ❌ Stack ensemble failed: {e}")
                
                # Bagging
                if 'bagging' in ensemble_types and len(base_models) > 0:
                    try:
                        logger.info("Creating Bagging ensemble...")
                        # 가장 좋은 성능의 모델 선택
                        best_base_model = base_models[0]
                        bagged_model = ensemble_model(best_base_model, method='Bagging', n_estimators=10, verbose=False)
                        all_ensemble_models[(model_key, area_grp_id, 'pycaret_bagging')] = bagged_model
                        
                        bag_results = pull().to_dict(orient="records")[0]
                        bag_results.update({
                            'model_key': model_key,
                            'area_grp_id': area_grp_id,
                            'model_name': 'pycaret_bagging',
                            'base_model': model_names_for_ensemble[0],
                            'params': 'bagging_n10'
                        })
                        all_model_results.append(bag_results)
                        logger.info(f"  ✅ Bagging ensemble MAE: {bag_results['MAE']:.4f}")
                    except Exception as e:
                        logger.info(f"  ❌ Bagging ensemble failed: {e}")
                
    # 결과 저장
    if enable_tuning:
        final_results_df = pd.DataFrame(all_grid_results)
        final_results_df["params"] = final_results_df["params"].astype(str)
        if category_name == 'wind':
            final_results_df['area_grp_id'] = final_results_df['area_grp_id'].apply(lambda x: x.split('_')[0])
        final_results_df = final_results_df[["area_grp_id", "model_name", "params", "MSE", "MAE", "MAPE", "RMSE", "R2", "RMSLE"]]
        aggregated_df = final_results_df.groupby(["params", "area_grp_id", "model_name"]).mean().reset_index()
        aggregated_df.to_csv(f'{save_dir}/train_tuned_results.csv', index=False)

    results_df = pd.DataFrame(all_model_results)
    results_df.to_csv(f'{save_dir}/train_results.csv', index=False)

    return all_trained_models, all_ensemble_models, feature_names_dict

def test_ml_pycaret(
    all_trained_models,
    all_ensemble_models,  # PyCaret 앙상블 모델들
    tst_loader,
    feature_names_dict,
    target_column="QGEN",
    save_dir="./saved_model/",
    eval_merge_grp=True,
    clamping=False,
    category_name='solar',
    enable_ensemble=True,  # 앙상블 메트릭 계산 여부
    return_output=False
):
    """
    Test function for individual models and optionally PyCaret ensemble models
    All CSV saving logic is maintained
    """
    from collections import defaultdict
    import numpy as np
    import pandas as pd
    import os
    from pycaret.regression import predict_model

    # 개별 모델 예측 수집
    region_preds_by_model = defaultdict(lambda: defaultdict(list))
    region_trues_by_model = defaultdict(lambda: defaultdict(list))
    
    # PyCaret 앙상블 예측 수집 (조건부 초기화)
    if enable_ensemble:
        pycaret_ensemble_preds = defaultdict(lambda: defaultdict(list))
        pycaret_ensemble_trues = defaultdict(lambda: defaultdict(list))

    # For saving all predictions and targets
    all_predictions_dict = {}  # key: (model_key, model_name), value: list of predictions
    all_targets_dict = {}  # key: (model_key, model_name), value: list of targets
    
    if enable_ensemble:
        ensemble_predictions_dict = {}  # key: (model_key, ensemble_type), value: list of predictions
        ensemble_targets_dict = {}  # key: (model_key, ensemble_type), value: list of targets

    # STEP 1: 개별 모델 예측
    logger.info("Testing individual models...")
    for (m_key, a_grp_id, model_name, params), trained_model in all_trained_models.items():
        model_key = m_key
        region = a_grp_id.split('_')[0]
        
        if model_key not in tst_loader or a_grp_id not in tst_loader[model_key]:
            continue
            
        data_dict = tst_loader[model_key][a_grp_id]
        test_x = data_dict["test_x"].drop(columns=["AREA_GRP_ID"], errors="ignore")
        test_y = data_dict["test_y"][target_column]

        # Feature alignment
        expected_features = feature_names_dict.get((model_key, a_grp_id), [])
        for feature in expected_features:
            if feature not in test_x.columns:
                test_x[feature] = 0
        extra_features = set(test_x.columns) - set(expected_features)
        if extra_features:
            test_x = test_x.drop(columns=list(extra_features))

        test_x = test_x[expected_features]
        test_data = pd.concat([test_x, test_y], axis=1)

        predictions = predict_model(trained_model, data=test_data, verbose=False)

        pred_values = predictions["prediction_label"].values
        true_values = predictions[target_column].values
        if clamping:
            pred_values = np.clip(pred_values, 0, None)

        region_preds_by_model[(model_key, model_name)][region].append(pred_values)
        region_trues_by_model[(model_key, model_name)][region].append(true_values)
        
        # Collect for output saving
        if return_output:
            key = (model_key, model_name)
            if key not in all_predictions_dict:
                all_predictions_dict[key] = []
                all_targets_dict[key] = []
            all_predictions_dict[key].append(pred_values)
            all_targets_dict[key].append(true_values)

    # STEP 1.5: PyCaret 앙상블 모델 예측 (조건부 실행)
    if enable_ensemble and all_ensemble_models:
        logger.info("Testing PyCaret ensemble models...")
        for (model_key, area_grp_id, ensemble_type), ensemble_model in all_ensemble_models.items():
            region = area_grp_id.split('_')[0]
            
            if model_key not in tst_loader or area_grp_id not in tst_loader[model_key]:
                continue
                
            data_dict = tst_loader[model_key][area_grp_id]
            test_x = data_dict["test_x"].drop(columns=["AREA_GRP_ID"], errors="ignore")
            test_y = data_dict["test_y"][target_column]

            expected_features = feature_names_dict.get((model_key, area_grp_id), [])
            test_x = test_x[expected_features]
            test_data = pd.concat([test_x, test_y], axis=1)

            predictions = predict_model(ensemble_model, data=test_data, verbose=False)
            pred_values = predictions["prediction_label"].values
            true_values = predictions[target_column].values
            
            if clamping:
                pred_values = np.clip(pred_values, 0, None)

            pycaret_ensemble_preds[(model_key, ensemble_type)][region].append(pred_values)
            pycaret_ensemble_trues[(model_key, ensemble_type)][region].append(true_values)
            
            # Collect for output saving
            if return_output:
                key = (model_key, ensemble_type)
                if key not in ensemble_predictions_dict:
                    ensemble_predictions_dict[key] = []
                    ensemble_targets_dict[key] = []
                ensemble_predictions_dict[key].append(pred_values)
                ensemble_targets_dict[key].append(true_values)

    # STEP 2: 메트릭 계산
    results = []
    area_level_metrics = []
    metric_cols = ['NMAE', 'MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE', 'RSE']

    # 개별 모델 메트릭
    logger.info("Calculating metrics for individual models...")
    for (model_key, model_name), region_dict in region_preds_by_model.items():
        for region in region_dict:
            num_areas = len(region_preds_by_model[(model_key, model_name)][region])
            
            if num_areas == 1:
                preds = region_preds_by_model[(model_key, model_name)][region][0]
                trues = region_trues_by_model[(model_key, model_name)][region][0]
                
                metrics = cal_metric(pred=preds, true=trues, region=region)
                results.append({
                    'model_key': model_key,
                    'model_name': model_name,
                    'model_type': 'individual',
                    'region': region,
                    'num_areas': 1,
                    'calculation_method': 'direct',
                    **metrics
                })
            else:
                area_metrics_list = []
                
                for i, (preds, trues) in enumerate(zip(
                    region_preds_by_model[(model_key, model_name)][region],
                    region_trues_by_model[(model_key, model_name)][region]
                )):
                    area_metrics = cal_metric(pred=preds, true=trues, region=region)
                    area_metrics_list.append(area_metrics)
                    
                    area_level_metrics.append({
                        'model_key': model_key,
                        'model_name': model_name,
                        'model_type': 'individual',
                        'region': region,
                        'area_index': i,
                        **area_metrics
                    })
                
                summed_metrics = {}
                for metric_name in metric_cols:
                    if metric_name in area_metrics_list[0]:
                        summed_metrics[metric_name] = sum(m[metric_name] for m in area_metrics_list)
                
                results.append({
                    'model_key': model_key,
                    'model_name': model_name,
                    'model_type': 'individual',
                    'region': region,
                    'num_areas': num_areas,
                    'calculation_method': 'summed',
                    **summed_metrics
                })
                
                logger.info(f"Individual model: {model_key}_{model_name}_{region} - Summed MAE: {summed_metrics.get('MAE', 0):.4f} (from {num_areas} areas)")

    # PyCaret 앙상블 메트릭 (조건부 실행)
    if enable_ensemble and all_ensemble_models:
        logger.info("Calculating metrics for PyCaret ensemble models...")
        for (model_key, ensemble_type), region_dict in pycaret_ensemble_preds.items():
            for region in region_dict:
                num_areas = len(pycaret_ensemble_preds[(model_key, ensemble_type)][region])
                
                if num_areas == 1:
                    preds = pycaret_ensemble_preds[(model_key, ensemble_type)][region][0]
                    trues = pycaret_ensemble_trues[(model_key, ensemble_type)][region][0]
                    
                    metrics = cal_metric(pred=preds, true=trues, region=region)
                    results.append({
                        'model_key': model_key,
                        'model_name': ensemble_type,
                        'model_type': 'pycaret_ensemble',
                        'region': region,
                        'num_areas': 1,
                        'calculation_method': 'direct',
                        **metrics
                    })
                else:
                    area_metrics_list = []
                    
                    for i, (preds, trues) in enumerate(zip(
                        pycaret_ensemble_preds[(model_key, ensemble_type)][region],
                        pycaret_ensemble_trues[(model_key, ensemble_type)][region]
                    )):
                        area_metrics = cal_metric(pred=preds, true=trues, region=region)
                        area_metrics_list.append(area_metrics)
                        
                        area_level_metrics.append({
                            'model_key': model_key,
                            'model_name': ensemble_type,
                            'model_type': 'pycaret_ensemble',
                            'region': region,
                            'area_index': i,
                            **area_metrics
                        })
                    
                    summed_metrics = {}
                    for metric_name in metric_cols:
                        if metric_name in area_metrics_list[0]:
                            summed_metrics[metric_name] = sum(m[metric_name] for m in area_metrics_list)
                    
                    results.append({
                        'model_key': model_key,
                        'model_name': ensemble_type,
                        'model_type': 'pycaret_ensemble',
                        'region': region,
                        'num_areas': num_areas,
                        'calculation_method': 'summed',
                        **summed_metrics
                    })
                    
                    logger.info(f"PyCaret {ensemble_type}: {model_key}_{region} - Summed MAE: {summed_metrics.get('MAE', 0):.4f} (from {num_areas} areas)")

    # STEP 3: 결과 저장 및 분석
    
    # 호기별 메트릭 저장
    if area_level_metrics:
        area_metrics_df = pd.DataFrame(area_level_metrics)
        area_metrics_df.to_csv(os.path.join(save_dir, "test_area_level_results.csv"), index=False)
    
    # 전체 결과 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(save_dir, "test_all_results.csv"), index=False)

    # 사용 가능한 메트릭 컬럼 확인
    available_metric_cols = [col for col in metric_cols if col in results_df.columns]

    # model_name과 region별 평균 계산
    averaged_results = (
        results_df
        .groupby(['model_name', 'region'], as_index=False)
        .agg({
            **{col: 'mean' for col in available_metric_cols},
            'model_key': lambda x: ','.join(sorted(x.unique())),
            'model_type': 'first',
            'num_areas': 'first',
            'calculation_method': 'first'
        })
        .rename(columns={'model_key': 'model_keys'})
    )

    averaged_results.to_csv(os.path.join(save_dir, "test_region_level_results.csv"), index=False)

    # 모델 타입별 성능 비교 (앙상블이 있는 경우에만)
    if enable_ensemble and 'pycaret_ensemble' in results_df['model_type'].values:
        model_type_comparison = (
            results_df
            .groupby(['model_type', 'region'], as_index=False)
            .agg({col: 'mean' for col in available_metric_cols})
        )
        
        model_type_comparison.to_csv(os.path.join(save_dir, "test_ensemble_model_comparison.csv"), index=False)
        
        # 개별 모델 vs PyCaret 앙상블 성능 비교
        comparison_results = []
        
        for region in results_df['region'].unique():
            region_results = results_df[results_df['region'] == region]
            
            # 개별 모델 중 최고 성능
            individual_results = region_results[region_results['model_type'] == 'individual']
            if len(individual_results) > 0:
                best_individual = individual_results.loc[individual_results['MAE'].idxmin()]
                
                # PyCaret 앙상블 성능
                ensemble_results = region_results[region_results['model_type'] == 'pycaret_ensemble']
                
                for _, ensemble_row in ensemble_results.iterrows():
                    comparison_results.append({
                        'region': region,
                        'best_individual_model': best_individual['model_name'],
                        'best_individual_MAE': best_individual['MAE'],
                        'ensemble_type': ensemble_row['model_name'],
                        'ensemble_MAE': ensemble_row['MAE'],
                        'improvement': best_individual['MAE'] - ensemble_row['MAE'],
                        'improvement_pct': ((best_individual['MAE'] - ensemble_row['MAE']) / best_individual['MAE'] * 100) if best_individual['MAE'] > 0 else 0
                    })
        
        if comparison_results:
            comparison_df = pd.DataFrame(comparison_results)
            comparison_df.to_csv(os.path.join(save_dir, "test_ensemble_comparison.csv"), index=False)
    
    # Save predictions and targets if requested
    if return_output:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save individual model predictions
        logger.info("Saving individual model predictions and targets...")
        for (model_key, model_name), pred_list in all_predictions_dict.items():
            total_outputs = np.concatenate(pred_list, axis=0)
            total_targets = np.concatenate(all_targets_dict[(model_key, model_name)], axis=0)
            
            model_filename = f"{model_key}_{model_name}"
            np.save(os.path.join(save_dir, f'TEST_{model_filename}_total_outputs.npy'), total_outputs)
            np.save(os.path.join(save_dir, f'TEST_{model_filename}_total_targets.npy'), total_targets)
            logger.info(f"Saved {model_filename}: {len(total_outputs)} predictions")
        
        # Save ensemble predictions if available
        if enable_ensemble and ensemble_predictions_dict:
            logger.info("Saving ensemble model predictions and targets...")
            for (model_key, ensemble_type), pred_list in ensemble_predictions_dict.items():
                total_outputs = np.concatenate(pred_list, axis=0)
                total_targets = np.concatenate(ensemble_targets_dict[(model_key, ensemble_type)], axis=0)
                
                ensemble_filename = f"{model_key}_{ensemble_type}"
                np.save(os.path.join(save_dir, f'TEST_{ensemble_filename}_total_outputs.npy'), total_outputs)
                np.save(os.path.join(save_dir, f'TEST_{ensemble_filename}_total_targets.npy'), total_targets)
                logger.info(f"Saved {ensemble_filename}: {len(total_outputs)} predictions")
            
    # 앙상블 사용 여부 로그
    if not enable_ensemble:
        logger.info("⚠️ Ensemble metrics were not calculated (enable_ensemble=False)")
    elif not all_ensemble_models:
        logger.info("⚠️ No ensemble models were provided")

    return averaged_results