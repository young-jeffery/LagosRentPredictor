# extract_tct_v2.py
# TCT is computed before spatial capping happens in this v2 run.

import pandas as pd
import re

def clean_base_rent(price_str):
    if pd.isna(price_str):
        return 0.0
    cleaned = re.sub(r'[^\d]', '', str(price_str))
    return float(cleaned) if cleaned else 0.0

def extract_fee(text, base_rent, fee_keywords):
    if pd.isna(text):
        return 0.0
    text = str(text).lower()
    keywords_joined = "|".join(fee_keywords)
    keywords_pattern = rf"\b(?:{keywords_joined})\b(?:\s+(?:fee|charge|deposit|fees))?(?:\s*\([^)]+\))?"
    separators = r"[\s\:\-\=\#n₦\?\*\|\>\@\_\.\;\u2013\u2014]*"

    perc_pattern = rf"{keywords_pattern}{separators}(\d+(?:\.\d+)?)%"
    perc_match = re.search(perc_pattern, text)
    if perc_match:
        return base_rent * (float(perc_match.group(1)) / 100.0)

    rev_perc_pattern = rf"(\d+(?:\.\d+)?)%\s*{keywords_pattern}"
    rev_perc_match = re.search(rev_perc_pattern, text)
    if rev_perc_match:
        return base_rent * (float(rev_perc_match.group(1)) / 100.0)

    abs_pattern = rf"{keywords_pattern}{separators}(?:ngn\s*|n\s*)?([\d\,\.]+)\s*([km]?)\b"
    abs_match = re.search(abs_pattern, text)
    if abs_match:
        val, mult = abs_match.group(1).replace(',', ''), abs_match.group(2)
        return calculate_absolute_value(val, mult, base_rent)

    rev_abs_pattern = rf"(?:ngn\s*|n\s*|\#)?([\d\,\.]+)\s*([km])\s+{keywords_pattern}"
    rev_abs_match = re.search(rev_abs_pattern, text)
    if rev_abs_match:
        val, mult = rev_abs_match.group(1).replace(',', ''), rev_abs_match.group(2)
        return calculate_absolute_value(val, mult, base_rent)

    return 0.0

def calculate_absolute_value(val_str, multiplier_str, base_rent):
    try:
        val = float(val_str)
        if multiplier_str == 'k':
            val *= 1000
        elif multiplier_str == 'm':
            val *= 1000000
        if val <= 100 and multiplier_str == '':
            return base_rent * (val / 100.0)
        return val
    except ValueError:
        return 0.0

def calculate_tct(input_csv, output_csv):
    print("--- STARTING TCT EXTRACTION (v2 - pre-cap) ---")
    df = pd.read_csv(input_csv)

    # [ADDED] This drop normally happens inside spatial_stratified_sampling.py.
    # In v2, capping runs LAST, so we do this filter here instead, right after load,
    # to keep behaviour equivalent to the original pipeline.
    before_drop = len(df)
    df = df[df['Main_District'] != 'Unknown']
    print(f"Dropped {before_drop - len(df)} rows with unmapped ('Unknown') location.")

    df['Combined_Text'] = df['Raw_Description'].fillna('') + " " + df['Detailed_Attributes'].fillna('')
    df['Base_Rent'] = df['Price'].apply(clean_base_rent)

    print("Extracting variable fees...")
    df['Fee_Agency'] = df.apply(lambda row: extract_fee(row['Combined_Text'], row['Base_Rent'], ['agency', 'agent']), axis=1)
    df['Fee_Legal'] = df.apply(lambda row: extract_fee(row['Combined_Text'], row['Base_Rent'], ['legal', 'agreement', 'agrement']), axis=1)
    df['Fee_Caution'] = df.apply(lambda row: extract_fee(row['Combined_Text'], row['Base_Rent'], ['caution', 'damage']), axis=1)
    df['Fee_Service'] = df.apply(lambda row: extract_fee(row['Combined_Text'], row['Base_Rent'], ['service charge', 'service', 'sc', 's/ charge', 's/charge']), axis=1)

    df['TCT'] = df['Base_Rent'] + df['Fee_Agency'] + df['Fee_Legal'] + df['Fee_Caution'] + df['Fee_Service']
    df = df.drop(columns=['Combined_Text'])

    df.to_csv(output_csv, index=False)
    print(f"Extraction complete. Output saved to '{output_csv}'")

if __name__ == "__main__":
    # [MODIFIED] Input is the raw mapped file (pre-cap), not the capped file.
    calculate_tct('lagos_rent_strictly_mapped.csv', 'lagos_rent_tct_extracted_v2.csv')