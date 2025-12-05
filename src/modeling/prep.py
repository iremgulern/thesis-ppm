import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def prepare_data(df):
    """
    Prepares features (X) and target (y) for modeling.
    Handles missing values, encoding, and feature assembly.
    """
    print("- Preparing data for modeling...")

    target_col = 'remaining_time_days'

    struct_cols = [c for c in df.columns if c.startswith('struct_count_')]
    seq_cols = [c for c in df.columns if c.startswith('seq_move_minus_')]
    workload_cols = ['judge_queue_length', 'dept_queue_length']

    numeric_cols = [
        'claim_amount', 'elapsed_time_days', 'time_since_last_event_days',
        'prefix_length', 'month', 'day_of_week', 'digital'
    ]

    base_categorical = ['movement', 'class', 'subject_matter', 'court_department']
    categorical_cols = base_categorical + seq_cols

    df = df.dropna(subset=[target_col]).copy()

    valid_num_cols = [c for c in numeric_cols + workload_cols + struct_cols if c in df.columns]
    df[valid_num_cols] = df[valid_num_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    valid_cat_cols = [c for c in categorical_cols if c in df.columns]
    df[valid_cat_cols] = df[valid_cat_cols].fillna('Unknown').astype(str)

    print(f"- Encoding {len(valid_cat_cols)} categorical columns...")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.int8)
    encoded_data = encoder.fit_transform(df[valid_cat_cols])
    encoded_feature_names = encoder.get_feature_names_out(valid_cat_cols).tolist()
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_feature_names, index=df.index)

    meta_cols = df[['lawsuit_id', 'case_start']]
    X = pd.concat([
        meta_cols,
        df[valid_num_cols],
        encoded_df
    ], axis=1)
    y = df[target_col]
    feature_names = valid_num_cols + encoded_feature_names

    return X, y, workload_cols, feature_names


def temporal_split(X, y, test_size=0.2):
    """
    Strict Temporal Split by Case.
    Sorts cases by start time, then takes the last 20% of CASES as the test set.
    Prevents data leakage between train/test.
    """
    print(f"- Splitting data (Strict Temporal, Test Size={test_size})...")

    case_starts = X.groupby('lawsuit_id')['case_start'].min().sort_values()
    split_idx = int(len(case_starts) * (1 - test_size))

    train_ids = set(case_starts.index[:split_idx])
    test_ids = set(case_starts.index[split_idx:])

    print(f"  Train Cases: {len(train_ids)} | Test Cases: {len(test_ids)}")
    mask_train = X['lawsuit_id'].isin(train_ids)
    mask_test = X['lawsuit_id'].isin(test_ids)

    cols_to_drop = ['lawsuit_id', 'case_start']
    X_train = X[mask_train].drop(columns=cols_to_drop)
    X_test = X[mask_test].drop(columns=cols_to_drop)
    y_train = y[mask_train]
    y_test = y[mask_test]

    return X_train, X_test, y_train, y_test