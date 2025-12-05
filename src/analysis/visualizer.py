import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shap
from src import config

# Ensure output directory exists
FIGURES_DIR = config.BASE_DIR / "reports" / "figures"


def _save_plot(filename):
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, bbox_inches='tight')
    plt.close()
    print(f"\tSaved: {FIGURES_DIR / filename}")


def _setup_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    if not FIGURES_DIR.exists():
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_duration_distribution(df):
    _setup_style()
    print("- Plotting Duration Distribution...")
    case_durations = df.groupby('lawsuit_id')['elapsed_time_days'].max()

    plt.figure(figsize=(10, 6))
    sns.histplot(case_durations, bins=50, kde=True, color='#3498db')
    plt.title('Distribution of Lawsuit Duration (Days)')
    plt.xlabel('Days')
    plt.ylabel('Number of Cases')
    _save_plot("01_duration_distribution.png")


def plot_workload_vs_duration(df):
    _setup_style()
    print("- Plotting Workload vs Duration (Scatter)...")

    # Aggregation
    case_stats = df.groupby('lawsuit_id').agg({
        'judge_queue_length': 'mean',
        'elapsed_time_days': 'max'
    })

    # Filter outliers (Top 5%)
    q95 = case_stats['elapsed_time_days'].quantile(0.95)
    df_plot = case_stats[case_stats['elapsed_time_days'] < q95]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_plot,
        x='judge_queue_length', y='elapsed_time_days',
        alpha=0.4, color='#2c3e50', edgecolor=None
    )
    plt.title('Hypothesis Check: Judge Workload vs. Lawsuit Duration')
    plt.xlabel('Average Judge Queue Size (Active Cases)')
    plt.ylabel('Case Duration (Days)')
    _save_plot("02_workload_vs_duration.png")


def plot_cases_per_judge(df):
    _setup_style()
    print("- Plotting Judge Caseload...")
    if 'judge' not in df.columns: return
    judge_counts = df.groupby('judge')['lawsuit_id'].nunique().sort_values(ascending=False).head(20)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=judge_counts.values, y=judge_counts.index, hue=judge_counts.index, legend=False, palette="viridis")
    plt.title('Top 20 Judges by Case Volume')
    plt.xlabel('Unique Lawsuits')
    _save_plot("03_cases_per_judge.png")


def plot_prefix_length_distribution(df):
    _setup_style()
    print("- Plotting Prefix Length Distribution...")
    if 'prefix_length' not in df.columns: return

    plt.figure(figsize=(10, 6))
    max_len = df['prefix_length'].quantile(0.99)
    data = df[df['prefix_length'] <= max_len]

    sns.histplot(data['prefix_length'], bins=30, kde=False, color='#9b59b6')
    plt.title('Distribution of Case Progress (Prefix Length)')
    plt.xlabel('Event Number (Prefix Length)')
    plt.ylabel('Frequency')
    _save_plot("05_prefix_length_distribution.png")


def plot_remaining_time_by_prefix(df):
    _setup_style()
    print("- Plotting Remaining Time vs Progress...")
    if 'prefix_length' not in df.columns or 'remaining_time_days' not in df.columns: return

    max_len = df['prefix_length'].quantile(0.95)
    df_plot = df[df['prefix_length'] <= max_len]
    mean_rem = df_plot.groupby('prefix_length')['remaining_time_days'].mean()

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=mean_rem.index, y=mean_rem.values, color='#e74c3c', linewidth=2.5)
    plt.title('Average Remaining Time by Case Progress')
    plt.xlabel('Event Number (Prefix Length)')
    plt.ylabel('Avg. Remaining Time (Days)')
    plt.grid(True, linestyle='--', linewidth=0.5)
    _save_plot("06_remaining_time_by_prefix.png")


def plot_shap_summary(model, X_test):
    _setup_style()
    print("- Calculating SHAP values...")

    # Subsample for speed
    X_sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
    if 'predicted_remaining' in X_sample.columns:
        X_sample = X_sample.drop(columns=['predicted_remaining'])

    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X_sample)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
        plt.title('SHAP Summary: Feature Impact on Duration')
        plt.savefig(FIGURES_DIR / "07_shap_summary.png")
        plt.close()
        print(f"\tSaved: {FIGURES_DIR / '07_shap_summary.png'}")
    except Exception as e:
        print(f"[!] SHAP failed: {e}")


def plot_error_by_prefix_length(X_test, y_test):
    _setup_style()
    print("- Plotting Error by Prefix Length...")

    df_eval = X_test.copy()
    df_eval['actual'] = y_test
    df_eval['error'] = abs(df_eval['predicted_remaining'] - df_eval['actual'])

    mae_by_len = df_eval.groupby('prefix_length')['error'].mean().head(20)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=mae_by_len.index, y=mae_by_len.values, marker='o', color='#c0392b', linewidth=2)
    plt.title('Model Error (MAE) by Case Progress')
    plt.xlabel('Event Number (Prefix Length)')
    plt.ylabel('Mean Absolute Error (Days)')
    plt.grid(True, linestyle='--')
    _save_plot("08_error_by_prefix.png")


def run_all_plots(df):
    plot_duration_distribution(df)
    plot_workload_vs_duration(df)
    plot_cases_per_judge(df)
    plot_prefix_length_distribution(df)
    plot_remaining_time_by_prefix(df)