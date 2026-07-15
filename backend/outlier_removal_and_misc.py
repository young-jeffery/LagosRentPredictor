import pandas as pd
import numpy as np

def hybrid_outlier_removal(input_file, output_file):
    print("--- INITIATING HYBRID OUTLIER FILTERING ---")

    df = pd.read_csv(input_file)
    initial_count = len(df)
    print(f"Total properties before cleaning: {initial_count}")

    # ============================================================
    # [ADDED] STEP 0: COMMERCIAL / NON-RESIDENTIAL FILTER
    #
    # The original pipeline only removed properties that were too
    # expensive. It did not catch listings that were a wrong property
    # TYPE entirely — warehouses priced per sqm, shortlets priced per
    # day, and industrial leases — where the TCT extractor pulled a
    # misleading low figure (e.g. ₦4,000 or ₦30,000) because it read
    # "per sqm" as the annual rent. These corrupt the model's location
    # signals: a Lekki warehouse appears to rent for ₦4k sitting next
    # to a real Lekki apartment at ₦8M. That contradiction cannot be
    # learned by any model.
    #
    # TO REVERT: Delete this entire Step 0 block (between the [ADDED]
    # and [END ADDED] markers). The df variable will still be defined
    # by pd.read_csv() above, so Step 1 will run cleanly without it.
    # ============================================================
    commercial_keywords = (
        r'warehouse|per sqm|per sq\.m|sqft|per sqft|'
        r'shortlet|short let|short-let|per day|per night|'
        r'daily rate|industrial estate|per square foot'
    )
    commercial_mask = df['Raw_Description'].str.contains(
        commercial_keywords, case=False, na=False, regex=True
    )

    # Hard floor: no genuine annual residential rent in Lagos is below
    # ₦150,000. Anything below this is a scraping artefact, a per-sqm
    # price, or a per-month figure the TCT extractor misread as annual.
    floor_mask = df['TCT'] < 150000

    combined_non_residential = commercial_mask | floor_mask
    non_residential_count = combined_non_residential.sum()
    df = df[~combined_non_residential].copy()
    print(f"[NEW] Step 0: Removed {non_residential_count} non-residential / implausible-floor rows.")
    print(f"              ({commercial_mask.sum()} matched commercial keywords, "
          f"{floor_mask.sum()} were below ₦150k floor, some rows matched both)")
    # ============================================================
    # [END ADDED]
    # ============================================================

    # ------------------------------------------------------------------
    # ORIGINAL HYBRID FILTER — unchanged from previous version.
    # Only difference: df is now the already-filtered frame from Step 0
    # instead of being freshly loaded. Behaviour is identical otherwise.
    # ------------------------------------------------------------------

    # 1. THE HARD CEILING (ORIGINAL)
    hard_ceiling = 40000000  # 40 Million Naira
    typo_mask = df['TCT'] > hard_ceiling
    typos_dropped = typo_mask.sum()
    df_step1 = df[~typo_mask]
    print(f"Step 1: Dropped {typos_dropped} properties exceeding the ₦40M Hard Ceiling.")

    # 2. THE TOP 5% CALCULATOR (ORIGINAL)
    p95_threshold = df_step1['TCT'].quantile(0.95)
    print(f"Step 2: The calculated Top 5% threshold is ₦{p95_threshold:,.2f}")

    # 3. THE SAFE OVERRIDE (ORIGINAL)
    safe_threshold = 15000000  # 15 Million Naira
    luxury_outlier_mask = (df_step1['TCT'] > p95_threshold) & (df_step1['TCT'] > safe_threshold)
    luxury_dropped = luxury_outlier_mask.sum()
    final_df = df_step1[~luxury_outlier_mask]

    if luxury_dropped > 0:
        print(f"Step 3: Dropped {luxury_dropped} luxury outliers strictly exceeding ₦{safe_threshold:,.2f}.")
    else:
        print(f"Step 3: The Top 5% fell below the safe threshold of ₦{safe_threshold:,.2f}. No valid properties were dropped.")

    final_count = len(final_df)
    total_dropped = initial_count - final_count
    print(f"\nFinal row count: {final_count} properties ready for training.")
    print(f"Total removed across all steps: {total_dropped}")

    final_df.to_csv(output_file, index=False)
    print(f"Cleaned dataset saved to '{output_file}'.")


if __name__ == "__main__":
    hybrid_outlier_removal('lagos_rent_tct_extracted.csv', 'lagos_rent_tct_cleaned.csv')




# ======================================================================
# ARCHIVED CODE — not part of the active pipeline
# ======================================================================

# GPU check
# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# if tf.test.is_built_with_cuda():
#     print("TensorFlow was built with CUDA (GPU support).")
# else:
#     print("TensorFlow was NOT built with CUDA.")

# Top rents viewer
# import pandas as pd
# df = pd.read_csv('lagos_rent_tct_cleaned.csv')
# print("--- TOP 10 HIGHEST RENTS IN DATASET ---")
# pd.options.display.float_format = '{:,.2f}'.format
# print(df[['Cleaned_Location', 'TCT']].sort_values(by='TCT', ascending=False).head(10))
