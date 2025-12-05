import pandas as pd


def add_judge_workload_features(df):
    """
    Calculates Judge and Dept workloads using optimized Interval Logic
    """
    df = df.sort_values(by=['date', 'order'])
    case_end_dates = df.groupby('lawsuit_id')['date'].max().rename('end_date')

    def get_workload_vectorized(entity_col):
        """
        Creates a +1/-1 timeline for every (Case, Entity) interaction.
        """

        starts = df.groupby(['lawsuit_id', entity_col])['date'].min().reset_index()
        starts = starts.dropna(subset=[entity_col])
        starts['change'] = 1

        ends = starts.copy()
        ends['date'] = ends['lawsuit_id'].map(case_end_dates)
        ends['date'] = ends['date'] + pd.Timedelta(microseconds=1)
        ends['change'] = -1
        timeline = pd.concat([starts, ends], ignore_index=True)
        timeline = timeline.sort_values(by='date')

        timeline['workload'] = timeline.groupby(entity_col)['change'].cumsum()
        merged = pd.merge_asof(
            df,
            timeline[['date', entity_col, 'workload']],
            on='date',
            by=entity_col,
            direction='backward'
        )

        return merged['workload'].fillna(0).astype(int)

    print("- Calculating Judge Workload (Vectorized)...")
    df['judge_queue_length'] = get_workload_vectorized('judge')

    print("- Calculating Department Workload (Vectorized)...")
    df['dept_queue_length'] = get_workload_vectorized('court_department')

    return df