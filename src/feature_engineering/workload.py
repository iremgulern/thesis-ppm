import pandas as pd
import numpy as np
from src import config


def add_inter_case_features(df):
    """
    Calculates Inter-case features (WIP by Section/State) and Judge Workload.
    """
    print("- Calculating Inter-case Features (WIP)...")
    df = df.sort_values(by=[config.COL_DATE, 'order'])

    # Case Lifespan
    case_spans = df.groupby(config.COL_CASE_ID)[config.COL_DATE].agg(['min', 'max']).reset_index()
    case_spans.columns = [config.COL_CASE_ID, 'start_date', 'end_date']

    def calculate_active_load(entity_col):
        if entity_col not in df.columns: return 0
        entity_map = df.drop_duplicates(config.COL_CASE_ID, keep='first')[[config.COL_CASE_ID, entity_col]]
        merged = case_spans.merge(entity_map, on=config.COL_CASE_ID).dropna(subset=[entity_col])

        starts = merged[['start_date', entity_col]].rename(columns={'start_date': config.COL_DATE})
        starts['change'] = 1
        ends = merged[['end_date', entity_col]].rename(columns={'end_date': config.COL_DATE})
        ends[config.COL_DATE] = ends[config.COL_DATE] + pd.Timedelta(seconds=1)
        ends['change'] = -1

        timeline = pd.concat([starts, ends]).sort_values(config.COL_DATE)
        timeline['load'] = timeline.groupby(entity_col)['change'].cumsum()

        merged_feat = pd.merge_asof(
            df.sort_values(config.COL_DATE),
            timeline[[config.COL_DATE, entity_col, 'load']],
            on=config.COL_DATE,
            by=entity_col,
            direction='backward'
        )
        return merged_feat['load'].fillna(0).astype(int)

    # 1. Judge Workload
    if config.COL_RESOURCE in df.columns:
        df['judge_workload'] = calculate_active_load(config.COL_RESOURCE)

    # 2. Section WIP (Inter-case)
    if 'court_department' in df.columns:
        df['section_wip'] = calculate_active_load('court_department')

    # 3. State WIP (Inter-case) - Approximated via Rolling Volume
    # Proxies 'busyness' of a specific movement/state
    daily_state_counts = df.groupby([config.COL_DATE, config.COL_ACTIVITY]).size().reset_index(name='daily_count')
    daily_state_counts = daily_state_counts.sort_values(config.COL_DATE)

    indexer = daily_state_counts.set_index(config.COL_DATE).groupby(config.COL_ACTIVITY)['daily_count'].rolling(
        '30D').sum().reset_index()

    df = pd.merge_asof(
        df.sort_values(config.COL_DATE),
        indexer.sort_values(config.COL_DATE),
        on=config.COL_DATE,
        by=config.COL_ACTIVITY,
        direction='backward'
    )
    df['state_load_30d'] = df['daily_count'].fillna(0)

    return df