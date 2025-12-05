import pandas as pd
from src import config

def clean_data(df):
    """
    Minimalist cleaner specific to the TJSP dataset structure.
    """
    df = df.copy()
    print("- Cleaning data (Vectorized)...")

    # Clean Currency: "17.284,24" -> 17284.24
    if 'claim_amount' in df.columns:
        df['claim_amount'] = (
            df['claim_amount']
            .astype(str)
            .str.replace('.', '', regex=False)
            .str.replace(',', '.', regex=False)
        )
        df['claim_amount'] = pd.to_numeric(df['claim_amount'], errors='coerce').fillna(0)

    # Clean Boolean: "VERDADEIRO" -> 1
    if 'digital' in df.columns:
        df['digital'] = (df['digital'].astype(str).str.upper().str.strip() == 'VERDADEIRO').astype(int)

    # Clean Dates: "01/06/2020" -> Datetime
    for col in config.DATE_COLS + config.DATETIME_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

    # Sorting (Essential for Process Mining: Case -> Date -> Order)
    sort_cols = [c for c in ['lawsuit_id', 'date', 'order'] if c in df.columns]
    if len(sort_cols) >= 2:
        df = df.sort_values(by=sort_cols)

    return df