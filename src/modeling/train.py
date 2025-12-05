import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from src.modeling.prep import temporal_split


def train_model(model_type, X_train, y_train):
    """
    Factory to create and train models with fixed hyperparameters.
    """
    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    elif model_type == 'xgb':
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, n_jobs=-1, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)
    return model


def run_experiment(X, y, workload_cols):
    """
    Runs a granular ablation study (8 scenarios) to isolate feature contributions.
    Tracks and returns the single best performing model for XAI.
    """
    X_train, X_test, y_train, y_test = temporal_split(X, y)

    results = []
    best_model = None
    X_test_best = None
    y_test_best = None
    min_mae = float('inf')
    all_cols = X_train.columns.tolist()

    feat_judge = ['judge_queue_length']
    feat_dept = ['dept_queue_length']
    feat_bag = [c for c in all_cols if c.startswith('struct_count_')]
    feat_seq = [c for c in all_cols if
                c.startswith('seq_move_minus_') or
                c.startswith('movement_') or
                c == 'prefix_length']
    feat_temp = ['elapsed_time_days', 'time_since_last_event_days', 'month', 'day_of_week']
    dynamic_feats = set(feat_judge + feat_dept + feat_bag + feat_seq + feat_temp)
    feat_static = [c for c in all_cols if c not in dynamic_feats]

    scenarios = {
        "1. Attributes Only": feat_static,
        "2. Temporal Only": feat_temp,
        "3. Control Flow (Seq)": feat_seq,
        "4. Control Flow (Bag)": feat_bag,
        "5. Baseline (Standard)": feat_static + feat_temp + feat_seq + feat_bag,
        "6. Baseline + Dept": feat_static + feat_temp + feat_seq + feat_bag + feat_dept,
        "7. Baseline + Judge": feat_static + feat_temp + feat_seq + feat_bag + feat_judge,
        "8. Extended (All)": all_cols
    }

    print("\n" + "=" * 40)
    print(f"STARTING ABLATION STUDY: {len(scenarios)} Scenarios")
    print("=" * 40)

    for model_name in ['rf', 'xgb']:
        print(f"\n- Model: {model_name.upper()}")

        for scenario_name, features in scenarios.items():
            if not features:
                print(f"  [Skip] {scenario_name} (No features)")
                continue

            model = train_model(model_name, X_train[features], y_train)
            y_pred = model.predict(X_test[features])

            mae = mean_absolute_error(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)

            print(f"  {scenario_name:25s} | MAE: {mae:.2f} days | RMSE: {rmse:.2f}")

            results.append({
                "Model": model_name.upper(),
                "Scenario": scenario_name,
                "MAE": mae,
                "RMSE": rmse,
                "Num_Features": len(features)
            })

            # Update best model
            if mae < min_mae:
                min_mae = mae
                best_model = model

                X_test_best = X_test[features].copy()
                X_test_best['predicted_remaining'] = y_pred
                y_test_best = y_test

                print(f"  -> New Best Model! (MAE: {min_mae:.2f})")

    return pd.DataFrame(results), best_model, X_test_best, y_test_best