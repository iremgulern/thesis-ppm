from src import config
from sklearn.preprocessing import StandardScaler

def target_encode(train_df, test_df, cat_cols, target, m=50):
    """Simple smoothing target encoder."""
    global_mean = train_df[target].mean()
    for col in cat_cols:
        stats = train_df.groupby(col)[target].agg(['mean', 'count'])
        smooth = (stats['count'] * stats['mean'] + m * global_mean) / (stats['count'] + m)
        mapping = smooth.to_dict()
        train_df[f"{col}_te"] = train_df[col].map(mapping).fillna(global_mean)
        test_df[f"{col}_te"] = test_df[col].map(mapping).fillna(global_mean)
    return train_df, test_df

def split_and_prepare_data(df):
    target = 'remaining_time_days'
    df = df.dropna(subset=[target]).copy()

    # Features
    num_cols = [c for c in df.columns if any(c.startswith(p) for p in ['Event_', 'Cluster_', 'judge_workload'])]
    num_cols += ['Elapsed_time', 'Time_since_last_event', 'prefix_length', 'judge_changed']
    cat_cols = ['Last_event_ID', 'Second_last_event_ID', 'Cluster'] + [a for a in config.CASE_ATTRIBUTES if a in df.columns]

    # Temporal Split
    cases = df.groupby(config.COL_CASE_ID)['case_start'].min().sort_values().index.tolist()
    split_idx = int(len(cases) * 0.7)
    train_df = df[df[config.COL_CASE_ID].isin(cases[:split_idx])].copy()
    test_df = df[df[config.COL_CASE_ID].isin(cases[split_idx:])].copy()

    # Encode & Scale
    train_df, test_df = target_encode(train_df, test_df, cat_cols, target)
    scaler = StandardScaler()
    train_df[num_cols] = scaler.fit_transform(train_df[num_cols].fillna(0))
    test_df[num_cols] = scaler.transform(test_df[num_cols].fillna(0))

    feature_names = num_cols + [f"{c}_te" for c in cat_cols]
    return {
        "X_train": train_df[[config.COL_CASE_ID] + feature_names].reset_index(drop=True),
        "y_train": train_df[target].reset_index(drop=True),
        "X_test": test_df[[config.COL_CASE_ID] + feature_names].reset_index(drop=True),
        "y_test": test_df[target].reset_index(drop=True),
        "feature_names": feature_names
    }