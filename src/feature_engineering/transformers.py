import pandas as pd
import numpy as np
from src import config
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


def add_temporal_features(df):
    """
    Temporal features: Weekday, Week number, Month number, Time since last, Elapsed.
    """
    df = df.copy()
    print("  - Adding Temporal Features...")

    if config.COL_DATE in df.columns:
        df[config.COL_DATE] = pd.to_datetime(df[config.COL_DATE])

    df = df.sort_values([config.COL_CASE_ID, config.COL_DATE])

    df['Weekday'] = df[config.COL_DATE].dt.dayofweek
    df['Month_number'] = df[config.COL_DATE].dt.month
    df['Week_number'] = df[config.COL_DATE].dt.isocalendar().week.astype(int)

    df['case_start'] = df.groupby(config.COL_CASE_ID)[config.COL_DATE].transform('min')
    df['Elapsed_time'] = (df[config.COL_DATE] - df['case_start']).dt.total_seconds().div(86400)
    df['Time_since_last_event'] = df.groupby(config.COL_CASE_ID)[config.COL_DATE].diff().dt.total_seconds().div(
        86400).fillna(0)

    # Target Variable (Remaining Time)
    # Note: For active cases, this is censored (Time until last observed event)
    df['case_end'] = df.groupby(config.COL_CASE_ID)[config.COL_DATE].transform('max')
    df['remaining_time_days'] = (df['case_end'] - df[config.COL_DATE]).dt.total_seconds().div(86400)

    return df


def add_control_flow_features(df, n_clusters=10, top_n_events=50):
    """
    1. Last Two Events
    2. Top 50 Frequent Events (Cumulative)
    3. Clustering (Co-occurrence + K-Means)
    """
    print(f"  - Adding Control Flow Features (Events={top_n_events}, Clusters={n_clusters})...")
    df = df.sort_values([config.COL_CASE_ID, config.COL_DATE])

    # 1. Last Two Events
    df['Last_event_ID'] = df.groupby(config.COL_CASE_ID)[config.COL_ACTIVITY].shift(1).fillna('Start')
    df['Second_last_event_ID'] = df.groupby(config.COL_CASE_ID)[config.COL_ACTIVITY].shift(2).fillna('Start')

    # 2. Top 50 Events (Cumulative Counts)
    event_counts = df[config.COL_ACTIVITY].value_counts()
    top_events = event_counts.head(top_n_events).index.tolist()

    # Create dummies
    dummies_events = pd.get_dummies(df[config.COL_ACTIVITY]).reindex(columns=top_events, fill_value=0)
    dummies_events.columns = [f"Event_{e}" for e in dummies_events.columns]

    # 3. Clustering
    print("    * Generating Co-occurrence Clusters...")
    unique_events = sorted(df[config.COL_ACTIVITY].unique())
    event_to_idx = {e: i for i, e in enumerate(unique_events)}
    n_unique = len(unique_events)
    df['move_id'] = df[config.COL_ACTIVITY].map(event_to_idx)

    # Build Co-occurrence Matrix
    matrix = np.zeros((n_unique, n_unique))
    traces = df.groupby(config.COL_CASE_ID)['move_id'].apply(list)
    for trace in traces:
        for i in range(len(trace) - 1):
            u, v = trace[i], trace[i + 1]
            matrix[u, v] += 1
            matrix[v, u] += 1

    # Cluster (K-Means)
    mat_norm = normalize(matrix, norm='l2', axis=1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(mat_norm)

    event_cluster_map = {event: str(label) for event, label in zip(unique_events, labels)}
    df['Cluster'] = df[config.COL_ACTIVITY].map(event_cluster_map).fillna('Unknown')

    # Cluster Dummies
    dummies_clusters = pd.get_dummies(df['Cluster'], prefix='Cluster')
    expected_clusters = [f"Cluster_{i}" for i in range(n_clusters)]
    dummies_clusters = dummies_clusters.reindex(columns=expected_clusters, fill_value=0)

    # Combine & Cumsum
    df = pd.concat([df, dummies_events, dummies_clusters], axis=1)
    cols_to_cumsum = dummies_events.columns.tolist() + expected_clusters
    df[cols_to_cumsum] = df.groupby(config.COL_CASE_ID)[cols_to_cumsum].cumsum()

    # Cleanup
    if 'move_id' in df.columns: del df['move_id']
    df['prefix_length'] = df.groupby(config.COL_CASE_ID).cumcount() + 1

    return df


def add_judge_change_feature(df):
    if config.COL_RESOURCE not in df.columns: return df

    print("  - Adding Judge Change Detection...")
    df = df.sort_values([config.COL_CASE_ID, config.COL_DATE])
    prev = df.groupby(config.COL_CASE_ID)[config.COL_RESOURCE].shift(1)
    df['judge_changed'] = ((df[config.COL_RESOURCE] != prev) & prev.notna()).astype(int)
    return df