import pandas as pd


def add_temporal_features(df):
    """
    Adds Temporal (elapsed, remaining, lag) + Prefix (length, context) features.
    """
    df = df.copy()
    df = df.sort_values(by=['lawsuit_id', 'date', 'order'])
    grp_date = df.groupby('lawsuit_id')['date']
    df['case_start'] = grp_date.transform('min')
    df['case_end'] = grp_date.transform('max')
    df['remaining_time_days'] = (df['case_end'] - df['date']).dt.total_seconds().div(86400)
    df['elapsed_time_days'] = (df['date'] - df['case_start']).dt.total_seconds().div(86400)
    prev_time = grp_date.shift(1)
    df['time_since_last_event_days'] = (df['date'] - prev_time).dt.total_seconds().div(86400).fillna(0)
    df['prefix_length'] = df.groupby('lawsuit_id').cumcount() + 1
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    return df


def add_structural_features(df, top_n=20):
    """
    Bag Abstraction: Cumulative counts of frequent activities.
    """
    print(f"- Generating Structural Features (Top {top_n} Bag)...")
    top_acts = df['movement'].value_counts().head(top_n).index
    dummies = pd.get_dummies(
        df['movement'].where(df['movement'].isin(top_acts)),
        dtype=int
    )
    struct_counts = dummies.groupby(df['lawsuit_id']).cumsum().add_prefix('struct_count_')
    return pd.concat([df, struct_counts], axis=1)


def add_sequence_features(df, history_len=3):
    """
    Index-Based Encoding: Captures order of last N events.
    """
    print(f"- Generating Sequence Features (History Len={history_len})...")
    df = df.copy()
    grouped_mov = df.groupby('lawsuit_id')['movement']

    for i in range(1, history_len + 1):
        df[f'seq_move_minus_{i}'] = grouped_mov.shift(i).fillna('Start')

    return df