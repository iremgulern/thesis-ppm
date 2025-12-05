def get_process_stats(df):
    """
    Calculates general process mining statistics.
    """
    print("- Calculating process stats...")
    stats = {
        "n_cases": df['lawsuit_id'].nunique(),
        "n_events": len(df),
        "n_activities": df['movement'].nunique(),
        "n_judges": df['judge'].nunique() if 'judge' in df.columns else 0
    }
    if 'remaining_time_days' in df.columns and 'elapsed_time_days' in df.columns:
        case_durations = df.groupby('lawsuit_id')['elapsed_time_days'].max()
    else:
        # Fallback: Timestamp difference
        grp = df.groupby('lawsuit_id')['date']
        case_durations = (grp.max() - grp.min()).dt.total_seconds().div(86400)

    stats["avg_duration_days"] = case_durations.mean()
    stats["median_duration_days"] = case_durations.median()

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
    print("\n" + "=" * 40)
    print("      DATASET STATISTICS")
    print("=" * 40)
    print(f"Cases:      {stats['n_cases']:,}")
    print(f"Events:     {stats['n_events']:,}")
    print(f"Activities: {stats['n_activities']}")
    print(f"Judges:     {stats['n_judges']}")
    print("-" * 40)
    print(f"Avg Duration:    {stats['avg_duration_days']:.2f} days")
    print(f"Median Duration: {stats['median_duration_days']:.2f} days")
    print("-" * 40)
    print("Top 3 Process Variants:")
    for i, (path, count) in enumerate(list(stats['top_variants'].items())[:3]):
        # Truncate string to 100 chars for clean display
        display_path = (path[:97] + '...') if len(path) > 100 else path
        print(f"{i + 1}. [{count} cases] {display_path}")
    print("=" * 40 + "\n")