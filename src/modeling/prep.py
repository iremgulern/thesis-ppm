import pandas as pd
import numpy as np
from src import config
from sklearn.preprocessing import StandardScaler


def target_encode_fit(train_df, cat_cols, target_col):
    mappings = {}
    global_mean = train_df[target_col].mean()
    for col in cat_cols:
        stats = train_df.groupby(col)[target_col].agg(['mean', 'count'])

        # [ADJUSTED] Lower smoothing for Control Flow to capture path-specific signals
        m = 2 if col in ['Last_event_ID', 'Cluster'] else 10

        smooth_mean = (stats['count'] * stats['mean'] + m * global_mean) / (stats['count'] + m)
        mappings[col] = smooth_mean.to_dict()
        mappings[col]['__global__'] = global_mean
    return mappings

def target_encode_transform(df, mappings, cat_cols):
    """
    Applies fitted Target Encoding mappings.
    """
    df_encoded = pd.DataFrame(index=df.index)
    for col in cat_cols:
        if col in mappings:
            global_val = mappings[col]['__global__']
            df_encoded[f"{col}_te"] = df[col].map(mappings[col]).fillna(global_val)
    return df_encoded

def split_and_prepare_data(df):
    print("- Starting Advanced Data Preparation (with Prefix Sampling)...")
    target_col = 'remaining_time_days'

    # [PAPER LOGIC] Ensure we are only training on prefixes of cases that actually have a remaining time
    df = df.dropna(subset=[target_col]).copy()

    # 1. Feature Groups (Same as before)
    cols_events = [c for c in df.columns if c.startswith('Event_')]
    cols_clusters = [c for c in df.columns if c.startswith('Cluster_') and c[8:].isdigit()]
    cols_intercase = [c for c in ['section_wip', 'state_load_30d'] if c in df.columns]
    cols_workload = ['judge_workload', 'judge_workload_ratio'] if 'judge_workload' in df.columns else []
    cols_temporal = ['Elapsed_time', 'Time_since_last_event', 'Month_number', 'Weekday', 'Week_number']

    cols_attrs_num = []
    cols_attrs_cat = []
    for attr in config.CASE_ATTRIBUTES:
        if attr in df.columns:
            if pd.api.types.is_numeric_dtype(df[attr]):
                cols_attrs_num.append(attr)
            else:
                cols_attrs_cat.append(attr)

    numeric_cols = cols_attrs_num + cols_temporal + cols_events + cols_clusters + cols_intercase + cols_workload + [
        'prefix_length', 'judge_changed']
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    cat_cols = ['Last_event_ID', 'Second_last_event_ID', 'Cluster'] + cols_attrs_cat
    cat_cols = [c for c in cat_cols if c in df.columns]

    # 2. Temporal Split
    case_starts = df.groupby(config.COL_CASE_ID)['case_start'].min().sort_values()
    cases = case_starts.index.tolist()
    n = len(cases)
    train_idx, val_idx = int(n * 0.70), int(n * 0.85)

    train_df = df[df[config.COL_CASE_ID].isin(cases[:train_idx])].copy()
    val_df = df[df[config.COL_CASE_ID].isin(cases[train_idx:val_idx])].copy()
    test_df = df[df[config.COL_CASE_ID].isin(cases[val_idx:])].copy()

    # [NEW] Data Balancing: If a case has 200 events, it might over-represent its path.
    # The paper uses "Strict Prefixes". We ensure we sample fairly.
    print(f"  - Target Encoding: {cat_cols}")
    te_mappings = target_encode_fit(train_df, cat_cols, target_col)

    for col in numeric_cols:
        mean_val = train_df[col].mean()
        for d in [train_df, val_df, test_df]: d[col] = d[col].fillna(mean_val)

    scaler = StandardScaler()
    scaler.fit(train_df[numeric_cols])

    def process(sub_df):
        if sub_df.empty: return None, None
        X_num = pd.DataFrame(scaler.transform(sub_df[numeric_cols]), columns=numeric_cols, index=sub_df.index)
        X_cat = target_encode_transform(sub_df, te_mappings, cat_cols)
        X = pd.concat([sub_df[[config.COL_CASE_ID, 'case_start']].reset_index(drop=True),
                       X_num.reset_index(drop=True),
                       X_cat.reset_index(drop=True)], axis=1)
        y = sub_df[target_col].reset_index(drop=True)
        return X, y

    return {
        "X_train": process(train_df)[0], "y_train": process(train_df)[1],
        "X_val": process(val_df)[0], "y_val": process(val_df)[1],
        "X_test": process(test_df)[0], "y_test": process(test_df)[1],
        "feature_names": numeric_cols + [f"{c}_te" for c in cat_cols]
    }