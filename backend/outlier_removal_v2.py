# outlier_removal_v2.py
# [MODIFIED] Step 3 (the luxury-outlier filter) has been rebuilt to use a
# PER-DISTRICT IQR fence instead of one flat global threshold.
#
# WHY: The old approach computed a single 95th-percentile + ₦15M safe
# override across the entire city. That punishes districts like Ikoyi/VI/
# Lekki for having naturally high prices relative to Lagos as a whole, and
# under-punishes budget districts, since a ₦15M "safe" override is a real
# anomaly in Ikorodu but not in Ikoyi. This version computes each
# district's own outlier fence from its own price distribution (Q3 + 3x
# its own IQR), so a genuinely typical high price in a premium district is
# no longer flagged, while a real anomaly in a budget district still is.
#
# Districts with fewer than MIN_SAMPLES_FOR_FENCE rows don't have enough
# data for a reliable IQR calculation on their own, so they fall back to
# a fence computed on the whole dataset instead.
#
# TO REVERT: Restore the old Step 3 block, commented out below the new one.

import pandas as pd
import numpy as np

# ============================================================
# [ADDED] Per-district IQR fence configuration
# TO REVERT: delete this block along with get_district_fence() below.
# ============================================================
MIN_SAMPLES_FOR_FENCE = 20   # Districts with fewer rows than this use the global fallback fence
FENCE_MULTIPLIER = 3.0       # Q3 + (multiplier * IQR) = upper fence. 3.0 = permissive "extreme outlier"
                              # standard, deliberately looser than the usual 1.5x "mild outlier" rule,
                              # since we want to keep legitimate high-end variance, not trim it.


def get_district_fence(group_df):
    """Computes the upper outlier fence for a single district's TCT values."""
    q1 = group_df['TCT'].quantile(0.25)
    q3 = group_df['TCT'].quantile(0.75)
    iqr = q3 - q1
    return q3 + FENCE_MULTIPLIER * iqr
# ============================================================
# [END ADDED]
# ============================================================


def hybrid_outlier_removal(input_file, output_file):
    print("--- INITIATING HYBRID OUTLIER FILTERING (v2 - per-district fence) ---")

    df = pd.read_csv(input_file)
    initial_count = len(df)
    print(f"Total properties before cleaning: {initial_count}")

    # Step 0: Commercial / non-residential filter (unchanged from before)
    commercial_keywords = (
        r'warehouse|per sqm|per sq\.m|sqft|per sqft|'
        r'shortlet|short let|short-let|per day|per night|'
        r'daily rate|industrial estate|per square foot'
    )
    commercial_mask = df['Raw_Description'].str.contains(
        commercial_keywords, case=False, na=False, regex=True
    )
    floor_mask = df['TCT'] < 150000

    combined_non_residential = commercial_mask | floor_mask
    non_residential_count = combined_non_residential.sum()
    df = df[~combined_non_residential].copy()
    print(f"Step 0: Removed {non_residential_count} non-residential / implausible-floor rows.")

    # Step 1: Global hard ceiling (unchanged — this catches absolute scraping
    # typos regardless of district, and stays global deliberately: no Lagos
    # residential rent should ever legitimately exceed this figure).
    hard_ceiling = 40000000
    typo_mask = df['TCT'] > hard_ceiling
    typos_dropped = typo_mask.sum()
    df = df[~typo_mask].copy()
    print(f"Step 1: Dropped {typos_dropped} properties exceeding the N40M Hard Ceiling.")

    # ============================================================
    # [MODIFIED] Step 3 (previously: global p95 + flat N15M override).
    # Now: per-district IQR fence, with a global fallback fence for
    # districts too small to compute a reliable fence of their own.
    # ============================================================
    district_counts = df['Main_District'].value_counts()
    reliable_districts = district_counts[district_counts >= MIN_SAMPLES_FOR_FENCE].index

    print(f"Step 2: Computing per-district fences for {len(reliable_districts)} districts "
          f"with >= {MIN_SAMPLES_FOR_FENCE} rows each.")
    print(f"        {(district_counts < MIN_SAMPLES_FOR_FENCE).sum()} small districts "
          f"will use the global fallback fence.")

    district_fences = {
        district: get_district_fence(df[df['Main_District'] == district])
        for district in reliable_districts
    }
    global_fallback_fence = get_district_fence(df)
    print(f"        Global fallback fence: N{global_fallback_fence:,.0f}")

    def exceeds_district_fence(row):
        fence = district_fences.get(row['Main_District'], global_fallback_fence)
        return row['TCT'] > fence

    luxury_outlier_mask = df.apply(exceeds_district_fence, axis=1)
    luxury_dropped = luxury_outlier_mask.sum()
    final_df = df[~luxury_outlier_mask].copy()
    print(f"Step 3: Dropped {luxury_dropped} properties exceeding their own district's fence.")
    # ============================================================
    # [END MODIFIED]
    # ============================================================

    # ------------------------------------------------------------
    # [REMOVED] Old global-threshold Step 2/3 — kept here for reference.
    # ------------------------------------------------------------
    # p95_threshold = df['TCT'].quantile(0.95)
    # safe_threshold = 15000000
    # luxury_outlier_mask = (df['TCT'] > p95_threshold) & (df['TCT'] > safe_threshold)
    # luxury_dropped = luxury_outlier_mask.sum()
    # final_df = df[~luxury_outlier_mask]
    # ------------------------------------------------------------

    final_count = len(final_df)
    total_dropped = initial_count - final_count
    print(f"\nFinal row count: {final_count} properties ready for capping.")
    print(f"Total removed across all steps: {total_dropped}")

    final_df.to_csv(output_file, index=False)
    print(f"Cleaned dataset saved to '{output_file}'.")


if __name__ == "__main__":
    hybrid_outlier_removal('lagos_rent_tct_extracted_v2.csv', 'lagos_rent_tct_cleaned_v2.csv')