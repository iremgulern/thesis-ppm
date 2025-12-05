import pandas as pd
from src import config

def load_data():
    """
    Loads raw data.
    """
    filepath = config.DATA_DIR / config.RAW_FILENAME
    print(f"- Loading: {filepath}")
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath, encoding='utf-8')
    print(f"- Loaded {len(df):,} rows.")
    return df


def save_data(df):
    """
    Saves processed data, ensuring the directory exists.
    """
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    filepath = config.DATA_PROCESSED_DIR / config.PROCESSED_FILENAME

    df.to_csv(filepath, index=False)
    print(f"- Saved processed data to: {filepath}")