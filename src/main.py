import pandas as pd
import config
from preprocessing import loader, cleaner, translator
from feature_engineering import transformers, workload
from analysis import visualizer, stats
from modeling import prep, train


def run_preprocessing():
    print("\n[1] Cleaning & Translation")
    p1_path = config.DATA_PROCESSED_DIR / config.TRANSLATED_FILENAME

    if p1_path.exists():
        print(f"- Found cached Phase 1: {p1_path.name}")
        df = pd.read_csv(p1_path)
        return df

    df = loader.load_data()
    df = cleaner.clean_data(df)
    df = translator.translate_data(df)

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(p1_path, index=False)
    return df


def run_feature_engineering(df_phase1):
    print("\n[2] Feature Engineering")
    p2_path = config.DATA_PROCESSED_DIR / config.FEATURED_FILENAME

    if p2_path.exists():
        print(f"- Checking cache: {p2_path.name}")
        df_feat = pd.read_csv(p2_path)

        has_seq = any(c.startswith('seq_move_minus_') for c in df_feat.columns)
        has_struct = any(c.startswith('struct_count_') for c in df_feat.columns)
        has_req = all(c in df_feat.columns for c in ['dept_queue_length', 'digital'])

        if has_seq and has_struct and has_req:
            print("- Cache valid. Loading...")
            for col in ['date', 'case_start', 'case_end']:
                if col in df_feat.columns:
                    df_feat[col] = pd.to_datetime(df_feat[col], errors='coerce')
            return df_feat

        print("! Cache stale or missing features. Regenerating...")

    if df_phase1 is None:
        raise ValueError("Phase 1 data missing.")

    print("- Generating Temporal & Structural Features...")
    df_feat = transformers.add_temporal_features(df_phase1)
    df_feat = transformers.add_structural_features(df_feat, top_n=20)
    df_feat = transformers.add_sequence_features(df_feat, history_len=3)

    print("- Calculating Workload (Judge & Dept)...")
    df_feat = workload.add_judge_workload_features(df_feat)

    df_feat.to_csv(p2_path, index=False)
    print(f"- Phase 2 saved to: {p2_path.name}")
    return df_feat


def run_analysis(df_feat):
    print("\n[3] Descriptive Analysis")
    process_stats = stats.get_process_stats(df_feat)
    stats.print_stats(process_stats)
    visualizer.run_all_plots(df_feat)


def run_modeling_and_explain(df_feat):
    print("\n[4] Predictive Modeling & XAI")
    X, y, workload_cols, feature_names = prep.prepare_data(df_feat)
    results_df, best_model, X_test, y_test = train.run_experiment(X, y, workload_cols)

    print("\n" + "=" * 40)
    print("FINAL RESULTS SUMMARY")
    print("=" * 40)
    print(results_df)

    if best_model is not None and X_test is not None:
        print("\n- Generating Advanced Visualizations...")
        visualizer.plot_error_by_prefix_length(X_test, y_test)
        visualizer.plot_shap_summary(best_model, X_test)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(config.MODEL_RESULTS_FILE, index=False)
    print(f"\nResults saved to {config.MODEL_RESULTS_FILE}")


def run_pipeline():
    print("=== PIPELINE START ===")
    df = run_preprocessing()
    df_feat = run_feature_engineering(df)
    run_analysis(df_feat)
    run_modeling_and_explain(df_feat)
    print("\n=== COMPLETE ===")


if __name__ == "__main__":
    run_pipeline()