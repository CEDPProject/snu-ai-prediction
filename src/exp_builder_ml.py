import numpy as np
from utils.utils import cal_metric, prepare_data

def training_ml(model_class, trn_loader):
    results = {}
    total_combinations = len(trn_loader)
    model_dict = {}

    for idx, ((model_key, area_grp_id), current_loader) in enumerate(trn_loader.items(), start=1):
        print(f"Processing combination: ({model_key}, {area_grp_id}) ({idx}/{total_combinations})")
        
        # Load or process data
        all_x, all_y = prepare_data(current_loader)
        
        print(f"Initializing model for ({model_key}, {area_grp_id})")
        # Initialize and save the model
        model_instance = model_class()
        model_dict[(model_key, area_grp_id)] = model_instance
        
        print(f"Fitting model for ({model_key}, {area_grp_id})")
        model_instance.mode.fit(all_x, all_y)

        print(f"Starting prediction")
        # Collect predictions
        all_predictions = []
        for batch in current_loader:
            x = batch['inputs']
            x_np = x.detach().cpu().numpy()
            predictions = model_instance.model.predict(x_np)
            all_predictions.append(predictions)
        
        all_predictions = np.concatenate(all_predictions)

        # Calculate metrics
        metrics = cal_metric(pred=all_predictions, true=all_y)

        print(f"Completed processing for ({model_key}, {area_grp_id})")
        print(f"metrics: {metrics}")

    return model_dict


def test_ml(model_dict, tst_loader):
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
        predictions = model_instance.model.predict(all_x)
        
        # Calculate metrics
        metrics = cal_metric(pred=predictions, true=all_y)
        print(f"Metrics for ({model_key}, {area_grp_id}): {metrics}")
        
        # Save results
        results[(model_key, area_grp_id)] = {
            'predictions': predictions,
            'metrics': metrics
        }

    return results