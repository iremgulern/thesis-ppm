import pandas as pd
import numpy as np
from src import config
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


def add_temporal_features(df):
    """
    Temporal features: Weekday, Week number, Month number, Time since last, Elapsed.
    Also calculates the target variable (remaining_time_days).
    """
    df = df.copy()
    print("  - Adding Temporal Features...")

    if config.COL_DATE in df.columns:
        df[config.COL_DATE] = pd.to_datetime(df[config.COL_DATE])

    df = df.sort_values([config.COL_CASE_ID, config.COL_DATE])

    # Basic time components
    df['Weekday'] = df[config.COL_DATE].dt.dayofweek
    df['Month_number'] = df[config.COL_DATE].dt.month
    df['Week_number'] = df[config.COL_DATE].dt.isocalendar().week.astype(int)

    # Durations within the case
    df['case_start'] = df.groupby(config.COL_CASE_ID)[config.COL_DATE].transform('min')
    df['Elapsed_time'] = (df[config.COL_DATE] - df['case_start']).dt.total_seconds().div(86400)
    df['Time_since_last_event'] = df.groupby(config.COL_CASE_ID)[config.COL_DATE].diff().dt.total_seconds().div(
        86400).fillna(0)

    # Target Variable: Remaining Time
    df['case_end'] = df.groupby(config.COL_CASE_ID)[config.COL_DATE].transform('max')
    df['remaining_time_days'] = (df['case_end'] - df[config.COL_DATE]).dt.total_seconds().div(86400)

    return df


def add_control_flow_features(df, n_clusters=10, top_n_events=50):
    """
    1. Sets current activity as the state (Last_event_ID).
    2. Calculates cumulative event counts (Process Memory).
    3. Groups transitions using a symmetric co-occurrence matrix.
    """
    print(f"  - Adding Control Flow Features (Events={top_n_events}, Clusters={n_clusters})...")
    df = df.sort_values([config.COL_CASE_ID, config.COL_DATE])
    case_col = config.COL_CASE_ID
    act_col = config.COL_ACTIVITY

    # 1. Immediate State (Technique from encode_last_two_event_ids)
    # Corrects the MAE by ensuring the model sees the CURRENT state.
    df['Last_event_ID'] = df[act_col]
    df['Second_last_event_ID'] = df.groupby(case_col)[act_col].shift(1)

    # 2. Cumulative Event Counts (Technique from encode_event_counts)
    event_counts = df[act_col].value_counts()
    top_events = event_counts.head(top_n_events).index.tolist()

    dummies_events = pd.get_dummies(df[act_col]).reindex(columns=top_events, fill_value=0)
    dummies_events.columns = [f"Event_{e}" for e in dummies_events.columns]

    # 3. Transition Clustering (Technique from build_cooccurrence_matrix & cluster_events)
    # Captures context by building a symmetric matrix of event adjacencies.
    unique_events = sorted(df[act_col].unique())
    e_to_i = {e: i for i, e in enumerate(unique_events)}
    matrix = np.zeros((len(unique_events), len(unique_events)))

    traces = df.groupby(case_col)[act_col].apply(lambda x: [e_to_i[ev] for ev in x])
    for trace in traces:
        for i in range(len(trace) - 1):
            u, v = trace[i], trace[i + 1]
            matrix[u, v] += 1
            matrix[v, u] += 1  # Ensure symmetry as used in the reference

    mat_norm = normalize(matrix, norm='l2', axis=1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    event_cluster_map = {ev: str(lab) for ev, lab in zip(unique_events, kmeans.fit_predict(mat_norm))}
    df['Cluster'] = df[act_col].map(event_cluster_map)

    # 4. Cluster Dummies (Technique from encode_cluster_counts)
    dummies_clusters = pd.get_dummies(df['Cluster'], prefix='Cluster')
    cluster_cols = [f"Cluster_{i}" for i in range(n_clusters)]
    dummies_clusters = dummies_clusters.reindex(columns=cluster_cols, fill_value=0)

    # 5. Apply Case-level Cumulative Sum
    # Transforms instantaneous events into a historical record.
    df = pd.concat([df, dummies_events, dummies_clusters], axis=1)
    cols_to_cumsum = dummies_events.columns.tolist() + cluster_cols
    df[cols_to_cumsum] = df.groupby(case_col)[cols_to_cumsum].cumsum()

    df['prefix_length'] = df.groupby(case_col).cumcount() + 1

    return df


def add_judge_change_feature(df):
    """
    Detects if the resource (judge) has changed compared to the previous event.
    """
    if config.COL_RESOURCE not in df.columns:
        return df

    print("  - Adding Judge Change Detection...")
    df = df.sort_values([config.COL_CASE_ID, config.COL_DATE])
    prev = df.groupby(config.COL_CASE_ID)[config.COL_RESOURCE].shift(1)
    df['judge_changed'] = ((df[config.COL_RESOURCE] != prev) & prev.notna()).astype(int)
    return df