# spatial_stratified_sampling_v2.py
# Same two-tier cap logic as spatial_stratified_sampling.py. Runs on the
# already-cleaned v2 file instead of the raw mapped file, so every cap
# slot is guaranteed to be a real, verified residential listing.
#
# [FIXED] The original version of this file used
#   df.groupby(col, group_keys=False).apply(lambda x: x.sample(...))
# which works fine on older pandas, but on pandas 3.0 (confirmed installed
# version), .apply() permanently strips the grouping column from both what
# the lambda receives and what it returns. This silently deleted
# 'Official_Sub_Location' after the first cap, and 'Main_District' after
# the second cap, from the saved output file. It did NOT affect which rows
# were sampled (verified by comparing row counts and cap constraints
# before/after the fix — both are identical), only which columns survived
# in the final CSV. Since we need these columns for future sub-location
# prediction work, this needed fixing.
#
# The fix replaces .apply() with a plain groupby loop + pd.concat, which
# does not trigger pandas 3.0's grouping-column exclusion behavior at all,
# because we're iterating the GroupBy object directly rather than passing
# a function into .apply().
#
# TO REVERT: Replace cap_group() calls with the old .groupby().apply()
# pattern shown commented out below. Not recommended — the old pattern
# loses Official_Sub_Location/Main_District on this pandas version.

import pandas as pd


def cap_group(df_in, group_col, cap, seed):
    """
    Samples up to `cap` rows per unique value of `group_col`, preserving
    every column including group_col itself. Replaces the old
    .groupby(col, group_keys=False).apply(lambda x: x.sample(...)) pattern,
    which drops group_col on pandas 3.0.
    """
    parts = []
    for _, group in df_in.groupby(group_col):
        n = min(len(group), cap)
        parts.append(group.sample(n, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def two_tier_stratified_sampling(input_csv, sub_cap=80, main_cap=250, seed=42):
    print("--- INITIATING SPATIAL CAPPING (v2 - post-cleaning, pandas-3.0-safe) ---")

    df = pd.read_csv(input_csv)
    print(f"Input rows (already cleaned): {len(df)}")

    # [MODIFIED] Uses cap_group() instead of .groupby().apply() to preserve
    # Official_Sub_Location and Main_District in the saved output.
    df_micro = cap_group(df, 'Official_Sub_Location', sub_cap, seed)
    print(f"After sub-location cap: {len(df_micro)} rows. "
          f"'Official_Sub_Location' present: {'Official_Sub_Location' in df_micro.columns}")

    df_balanced = cap_group(df_micro, 'Main_District', main_cap, seed)
    print(f"After district cap: {len(df_balanced)} rows. "
          f"'Main_District' present: {'Main_District' in df_balanced.columns}")

    # ------------------------------------------------------------
    # [REMOVED] Old .apply()-based version — kept here for reference only.
    # Do not use on pandas >= 2.2, it silently drops the grouping column.
    # ------------------------------------------------------------
    # df_micro = (df.groupby('Official_Sub_Location', group_keys=False)
    #               .apply(lambda x: x.sample(min(len(x), sub_cap), random_state=seed)))
    # df_balanced = (df_micro.groupby('Main_District', group_keys=False)
    #                        .apply(lambda x: x.sample(min(len(x), main_cap), random_state=seed)))
    # ------------------------------------------------------------

    print(f"\nBalanced Dataset Size: {len(df_balanced)} rows.")

    output_name = 'lagos_rent_spatially_capped_v2.csv'
    df_balanced.to_csv(output_name, index=False)
    print(f"Success. Saved to '{output_name}'.")
    print(f"Columns in saved file: {df_balanced.columns.tolist()}")


if __name__ == "__main__":
    two_tier_stratified_sampling('lagos_rent_tct_cleaned_v2.csv', sub_cap=80, main_cap=250)
