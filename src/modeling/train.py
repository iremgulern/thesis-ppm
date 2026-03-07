import pandas as pd
from src import config
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def train_model(model_type, X_train, y_train):
    """
    Fits the specified model on the training data.
    """
    if model_type == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    elif model_type == 'xgb':
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)
    return model


def run_experiment(data_dict):
    """
    Runs an ablation study across different feature families to evaluate MAE.
    Logic updated to ensure Control Flow scenarios capture the current process state.
    """
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    all_cols = data_dict['feature_names']

    # Drop non-feature metadata
    meta_cols = [config.COL_CASE_ID, 'case_start']
    X_train_cl = X_train.drop(columns=meta_cols, errors='ignore')
    X_test_cl = X_test.drop(columns=meta_cols, errors='ignore')

    # --- Define Feature Families ---
    # Case Attributes (Baseline)
    f_attrs = [c for c in all_cols if any(attr in c for attr in config.CASE_ATTRIBUTES) and '_te' in c]

    # Control Flow: Cumulative Counts
    f_events = [c for c in all_cols if c.startswith('Event_')]
    f_clusters = [c for c in all_cols if c.startswith('Cluster_')]

    # Control Flow: Current State IDs (Target Encoded)
    f_state = ['Last_event_ID_te']
    f_last_two = ['Last_event_ID_te', 'Second_last_event_ID_te']

    # Other groups
    f_workload = [c for c in all_cols if 'workload' in c]
    f_temporal = ['Elapsed_time', 'Time_since_last_event', 'Month_number_te', 'Weekday_te', 'Week_number_te']

    def get_cols(col_list):
        valid = [c for c in col_list if c in X_train_cl.columns]
        return sorted(list(set(valid)))

    # --- Define Scenarios ---
    scenarios = {
        "1. Case Attributes (Baseline)": get_cols(f_attrs),
        "2. Control Flow: Events (Freq + State)": get_cols(f_attrs + f_events + f_state),
        "3. Control Flow: Clusters": get_cols(f_attrs + f_clusters + ['Cluster_te']),
        "4. Control Flow: Last Two": get_cols(f_attrs + f_last_two),
        "5. Full Control Flow": get_cols(f_attrs + f_events + f_clusters + f_last_two),
        "6. Temporal Features": get_cols(f_attrs + f_temporal),
        "7. Workload Features": get_cols(f_attrs + f_workload),
        "8. All Features": X_train_cl.columns.tolist()
    }

    results = []
    best_model, min_mae = None, float('inf')
    X_test_best, y_test_best = None, None

    print("\n" + "=" * 60)
    print("=" * 60)

    for model_name in ['rf', 'xgb']:
        print(f"\n- Model: {model_name.upper()}")
        for scenario_name, features in scenarios.items():
            if not features:
                continue

            model = train_model(model_name, X_train_cl[features], y_train)
            y_pred = model.predict(X_test_cl[features])

            mae = mean_absolute_error(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)

            print(f"  {scenario_name:35s} | MAE: {mae:.2f}")

            results.append({
                "Model": model_name.upper(),
                "Scenario": scenario_name,
                "MAE": mae,
                "RMSE": rmse,
                "Num_Features": len(features)
            })

            if mae < min_mae:
                min_mae = mae
                best_model = model
                X_test_best, y_test_best = X_test.copy(), y_test
                X_test_best['predicted_remaining'] = y_pred
                print(f"  -> New Best Model! (MAE: {min_mae:.2f})")

    return pd.DataFrame(results), best_model, X_test_best, y_test_best