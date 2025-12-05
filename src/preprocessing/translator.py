import json
import time
import re
from deep_translator import GoogleTranslator
from src import config

CLEAN_REGEX = re.compile(r'[\r\n\t\u2028\u2029]+')

def _sanitize_text(text):
    """
    Removes newlines and control characters from translated text.
    """
    if not isinstance(text, str):
        return text
    return CLEAN_REGEX.sub(' ', text).strip()


def _load_cache():
    """Loads translation cache JSON."""
    if config.TRANSLATION_CACHE_FILE.exists():
        try:
            with open(config.TRANSLATION_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("! Warning: Corrupt translation cache. Starting fresh.")
    return {}


def _save_cache(cache):
    """Saves translation cache JSON."""
    with open(config.TRANSLATION_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=4, ensure_ascii=False)


def _translate_terms(terms_list):
    """
    Translates a batch of terms using Google Translate with rate limiting.
    """
    translator = GoogleTranslator(source='pt', target='en')
    results = {}
    print(f"- Translating {len(terms_list)} new terms...")

    for i, term in enumerate(terms_list):
        if not term or str(term).strip() == "":
            results[term] = term
            continue
        try:
            # Respect API rate limits
            time.sleep(0.2)
            trans = translator.translate(term)
            results[term] = _sanitize_text(trans)
        except Exception as e:
            print(f"  ! Error translating '{term}': {e}")
            results[term] = term  # Fallback to original

    return results


def _resolve_collisions(mapping_dict):
    """
    Ensures 1-to-1 mapping to preserve process variant distinctness.
    If two Portuguese terms map to 'Petition', the second becomes 'Petition (Petição X)'.
    """
    reverse_map = {}
    final_map = {}

    for pt, en in mapping_dict.items():
        clean_en = str(en).strip()

        if clean_en not in reverse_map:
            reverse_map[clean_en] = pt
            final_map[pt] = clean_en
        else:
            # Collision detected
            final_map[pt] = f"{clean_en} ({pt})"

    return final_map


def translate_data(df):
    """
    Orchestrates the translation of categorical columns using a persistent cache.
    """
    df = df.copy()
    cache = _load_cache()
    updated = False

    print("- Translating categorical columns...")

    for col in config.CATEGORICAL_COLS:
        if col not in df.columns:
            continue

        unique_vals = df[col].dropna().unique()
        if col not in cache:
            cache[col] = {}
        missing = [v for v in unique_vals if v not in cache[col]]

        if missing:
            new_trans = _translate_terms(missing)
            cache[col].update(new_trans)
            updated = True

        cache[col] = _resolve_collisions(cache[col])
        df[col] = df[col].map(cache[col]).fillna(df[col])

    if updated:
        _save_cache(cache)
        print(f"- Updated translation cache at: {config.TRANSLATION_CACHE_FILE}")

    return df