import pandas as pd

def get_process_stats(df):
    """
    Calculates detailed process mining statistics including min/max/median/mean.
    """
    print("- Calculating process stats...")

    # Identify Duration Column
    dur_col = 'elapsed_time_days'
    if 'Elapsed_time' in df.columns:
        dur_col = 'Elapsed_time'

    # Group by Case
    if 'remaining_time_days' in df.columns and dur_col in df.columns:
        case_durations = df.groupby('lawsuit_id')[dur_col].max()
    else:
        if 'date' in df.columns:
            grp_date = df.groupby('lawsuit_id')['date']
            case_durations = (grp_date.max() - grp_date.min()).dt.total_seconds().div(86400)
        else:
            case_durations = pd.Series(dtype=float)

    case_lengths = df.groupby('lawsuit_id').size()

    stats = {
        "n_cases": df['lawsuit_id'].nunique(),
        "n_events": len(df),
        "n_activities": df['movement'].nunique(),
        "n_judges": df['judge'].nunique() if 'judge' in df.columns else 0,

        # Duration Stats (Throughput Time)
        "duration_min": case_durations.min() if not case_durations.empty else 0,
        "duration_max": case_durations.max() if not case_durations.empty else 0,
        "duration_mean": case_durations.mean() if not case_durations.empty else 0,
        "duration_median": case_durations.median() if not case_durations.empty else 0,

        # Case Length Stats
        "length_min": case_lengths.min(),
        "length_max": case_lengths.max(),
        "length_mean": case_lengths.mean(),
        "length_median": case_lengths.median(),
    }

    # Remaining time stats
    if 'remaining_time_days' in df.columns:
        stats['rem_time_min'] = df['remaining_time_days'].min()
        stats['rem_time_max'] = df['remaining_time_days'].max()
        stats['rem_time_mean'] = df['remaining_time_days'].mean()
        stats['rem_time_median'] = df['remaining_time_days'].median()

    # Variants
    sort_cols = [c for c in ['lawsuit_id', 'date', 'order'] if c in df.columns]
    df_sorted = df.sort_values(by=sort_cols)
    variants = df_sorted.groupby('lawsuit_id')['movement'].agg(tuple)
    top_variants_tuples = variants.value_counts().head(5)
    stats["top_variants"] = {
        " -> ".join(map(str, var_tuple)): count
        for var_tuple, count in top_variants_tuples.items()
    }

    return stats


def print_stats(stats):
    """Pretty prints the statistics."""
    print("\n" + "=" * 50)
    print("      DATASET STATISTICS")
    print("=" * 50)
    print(f"Cases:      {stats['n_cases']:,}")
    print(f"Events:     {stats['n_events']:,}")
    print(f"Activities: {stats['n_activities']}")
    print(f"Judges:     {stats['n_judges']}")
    print("-" * 50)
    print("CASE LENGTH (Events per Case):")
    print(f"  Min: {stats['length_min']}")
    print(f"  Max: {stats['length_max']}")
    print(f"  Mean: {stats['length_mean']:.2f}")
    print(f"  Median: {stats['length_median']:.2f}")
    print("-" * 50)
    print("THROUGHPUT TIME (Days):")
    print(f"  Min: {stats['duration_min']:.2f}")
    print(f"  Max: {stats['duration_max']:.2f}")
    print(f"  Mean: {stats['duration_mean']:.2f}")
    print(f"  Median: {stats['duration_median']:.2f}")

    if 'rem_time_mean' in stats:
        print("-" * 50)
        print("REMAINING TIME (Target - Days):")
        print(f"  Min: {stats['rem_time_min']:.2f}")
        print(f"  Max: {stats['rem_time_max']:.2f}")
        print(f"  Mean: {stats['rem_time_mean']:.2f}")
        print(f"  Median: {stats['rem_time_median']:.2f}")

    print("-" * 50)
    print("Top 3 Process Variants:")
    for i, (path, count) in enumerate(list(stats['top_variants'].items())[:3]):
        display_path = (path[:97] + '...') if len(path) > 100 else path
        print(f"{i + 1}. [{count} cases] {display_path}")
    print("=" * 50 + "\n")