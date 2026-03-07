import pandas as pd
from src import config
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def train_model(model_type, X_train, y_train):
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
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    all_cols = data_dict['feature_names']

    meta_cols = [config.COL_CASE_ID, 'case_start']
    X_train_cl = X_train.drop(columns=meta_cols, errors='ignore')
    X_test_cl = X_test.drop(columns=meta_cols, errors='ignore')

    # Feature Families
    f_attrs = [c for c in all_cols if
               any(attr in c for attr in config.CASE_ATTRIBUTES) and '_te' in c or c in config.CASE_ATTRIBUTES]
    f_events = [c for c in all_cols if c.startswith('Event_')]
    f_clusters = [c for c in all_cols if c.startswith('Cluster_')]
    f_last_two = ['Last_event_ID_te', 'Second_last_event_ID_te']
    f_intercase = ['section_wip', 'state_load_30d']
    f_workload = ['judge_workload', 'judge_workload_ratio']
    f_temporal = ['Weekday_te', 'Week_number_te', 'Month_number_te', 'Time_since_last_event', 'Elapsed_time']

    def get_cols(col_list):
        valid = [c for c in col_list if c in X_train_cl.columns]
        return sorted(list(set(valid)))

    scenarios = {
        "1. Case Attributes (Baseline)": get_cols(f_attrs),
        "2. Control Flow: Events (Freq)": get_cols(f_attrs + f_events),
        "3. Control Flow: Clusters": get_cols(f_attrs + f_clusters),
        "4. Control Flow: Last Two": get_cols(f_attrs + f_last_two),
        "5. State Features": get_cols(f_attrs + f_events),  # Paper uses events/states interchangeably
        "6. Inter-case Features": get_cols(f_attrs + f_intercase),
        "7. Judge Workload": get_cols(f_attrs + f_workload),
        "8. Temporal Features": get_cols(f_attrs + f_temporal),
        "9. All Features": X_train_cl.columns.tolist()
    }

    results = []
    best_model, min_mae = None, float('inf')
    X_test_best, y_test_best = None, None

    print("\n" + "=" * 60)
    print(f"STARTING COMPREHENSIVE ABLATION (Prefix Sampling Enabled)")
    print("=" * 60)

    for model_name in ['rf', 'xgb']:
        print(f"\n- Model: {model_name.upper()}")
        for scenario_name, features in scenarios.items():
            if not features: continue
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