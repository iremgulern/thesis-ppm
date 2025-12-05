from pathlib import Path

# Project Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# File Paths
RAW_FILENAME = "TJSP-BL-event-log.csv"
TRANSLATED_FILENAME = "tjsp_translated.csv"
FEATURED_FILENAME = "tjsp_features.csv"
TRANSLATION_CACHE_FILE = DATA_DIR / "translation_cache.json"
MODEL_RESULTS_FILE = REPORTS_DIR / "model_results.csv"

# Data Schema
CATEGORICAL_COLS = ['movement', 'status', 'class', 'subject_matter', 'court_department']
CURRENCY_COLS = ['claim_amount']
DATE_COLS = ['date']
DATETIME_COLS = ['distribution_date']

# Formats
DATE_FORMAT = "%d/%m/%Y"