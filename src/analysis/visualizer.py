import matplotlib.pyplot as plt
import seaborn as sns
from src import config


def run_all_plots(df):
    print("- Generating Plots...")
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_duration_distribution(df)

    if 'judge_queue_length' in df.columns:
        plot_workload_vs_duration(df)

    if 'judge' in df.columns:
        plot_cases_per_judge(df)

    if 'prefix_length' in df.columns:
        plot_prefix_length_distribution(df)

    if 'remaining_time_days' in df.columns and 'prefix_length' in df.columns:
        plot_remaining_time_by_prefix(df)


def plot_duration_distribution(df):
    plt.figure(figsize=(10, 6))

    dur_col = 'elapsed_time_days'
    if 'Elapsed_time' in df.columns:
        dur_col = 'Elapsed_time'

    if dur_col not in df.columns:
        plt.close()
        return

    case_durations = df.groupby('lawsuit_id')[dur_col].max()

    sns.histplot(case_durations, bins=50, kde=True)
    plt.title('Case Duration Distribution')
    plt.xlabel('Days')
    plt.ylabel('Count')
    plt.savefig(config.FIGURES_DIR / '01_duration_distribution.png')
    plt.close()


def plot_workload_vs_duration(df):
    plt.figure(figsize=(10, 6))

    dur_col = 'elapsed_time_days'
    if 'Elapsed_time' in df.columns:
        dur_col = 'Elapsed_time'

    if dur_col not in df.columns:
        plt.close()
        return

    plot_df = df.sample(n=min(10000, len(df)), random_state=42)

    sns.scatterplot(data=plot_df, x='judge_queue_length', y=dur_col, alpha=0.3)
    plt.title('Judge Workload vs. Elapsed Time')
    plt.xlabel('Judge Queue Length')
    plt.ylabel('Elapsed Time (Days)')
    plt.savefig(config.FIGURES_DIR / '02_workload_vs_duration.png')
    plt.close()


def plot_cases_per_judge(df):
    plt.figure(figsize=(12, 6))
    top_judges = df['judge'].value_counts().head(20)
    sns.barplot(x=top_judges.index, y=top_judges.values)
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 20 Judges by Event Volume')
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / '03_cases_per_judge.png')
    plt.close()


def plot_prefix_length_distribution(df):
    plt.figure(figsize=(10, 6))
    case_lengths = df.groupby('lawsuit_id')['prefix_length'].max()
    sns.histplot(case_lengths, bins=30, kde=False)
    plt.title('Distribution of Case Lengths (Events per Case)')
    plt.xlabel('Number of Events')
    plt.savefig(config.FIGURES_DIR / '05_prefix_length_distribution.png')
    plt.close()


def plot_remaining_time_by_prefix(df):
    plt.figure(figsize=(10, 6))
    avg_rem = df.groupby('prefix_length')['remaining_time_days'].mean().reset_index()
    avg_rem = avg_rem[avg_rem['prefix_length'] <= 50]

    sns.lineplot(data=avg_rem, x='prefix_length', y='remaining_time_days')
    plt.title('Average Remaining Time by Case Progress')
    plt.xlabel('Prefix Length (Event #)')
    plt.ylabel('Avg Remaining Time (Days)')
    plt.savefig(config.FIGURES_DIR / '06_remaining_time_by_prefix.png')
    plt.close()


def plot_error_by_prefix_length(X_test, y_test):
    if 'prefix_length' not in X_test.columns or 'predicted_remaining' not in X_test.columns:
        return

    df_eval = X_test.copy()
    df_eval['actual'] = y_test
    df_eval['abs_error'] = abs(df_eval['actual'] - df_eval['predicted_remaining'])

    df_eval['prefix_capped'] = df_eval['prefix_length'].apply(lambda x: x if x <= 40 else 40)

    mae_by_prefix = df_eval.groupby('prefix_capped')['abs_error'].mean()

    plt.figure(figsize=(10, 6))
    mae_by_prefix.plot(kind='line', marker='o')
    plt.title('MAE by Case Progress')
    plt.xlabel('Prefix Length (Capped at 40)')
    plt.ylabel('Mean Absolute Error (Days)')
    plt.grid(True)
    plt.savefig(config.FIGURES_DIR / '08_error_by_prefix.png')
    plt.close()