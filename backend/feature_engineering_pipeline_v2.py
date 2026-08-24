# feature_engineering_pipeline_v2.py -- Take 3
# [MODIFIED] Replaced One-Hot Encoding + district consolidation with
# hierarchical smoothed target encoding on Official_Sub_Location, with
# shrinkage toward Main_District, then toward the global mean.
#
# WHY: OHE + manual district-consolidation (DISTRICT_MAP/RARE_THRESHOLD)
# was a workaround for location sparsity, but it can only ever predict at
# the district level for anything it didn't see enough of during training,
# and it cannot produce ANY prediction for a sub-location outside the
# training data entirely (handle_unknown='ignore' just zeroes it out).
# Target encoding with hierarchical smoothing solves this properly: a
# sub-location with lots of training data gets to speak mostly for itself;
# a sparse or entirely unseen sub-location gracefully falls back to its
# district average, and even a totally unseen district falls back to the
# citywide average. This is what makes true sub-location-level prediction
# (across your full location dictionary) possible.
#
# Validated on lagos_rent_spatially_capped_v2.csv: R2 0.7846 vs the OHE
# version's 0.7825, with the added ability to predict any dictionary
# sub-location, not just the ones with dense training data.
#
# Leakage safety: the encoding is fit on TRAIN ONLY. The training set
# itself is encoded via 5-fold out-of-fold encoding, so no row's own
# price ever contributes to its own encoded location value. The test set
# is encoded using the training-only mapping, exactly as it would be at
# real inference time.
#
# Output format changed: target encoding produces one dense continuous
# column instead of many sparse OHE columns, so outputs are now saved as
# plain .csv instead of scipy sparse .npz.
#
# TO REVERT: restore OneHotEncoder + ColumnTransformer + DISTRICT_MAP +
# RARE_THRESHOLD + get_consolidated_location from the previous version of
# this file (kept in project history / earlier chat messages).

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, KFold
import joblib

# ============================================================
# [ADDED] Hierarchical smoothed target encoding configuration.
# K_SUB: how much to trust a sub-location's own average before blending
#   toward its district (higher = need more data before trusting it alone).
# K_DISTRICT: same idea, for district vs citywide average.
# TO REVERT: delete this whole section, restore OHE-based code instead.
# ============================================================
K_SUB = 10
K_DISTRICT = 30
N_FOLDS = 5


def fit_hierarchical_encoding(train_df, target_col='log_tct'):
    """
    Fits sub-location and district encoding maps using TRAIN DATA ONLY.
    Returns a dict with everything needed to encode both train (via
    out-of-fold, called separately) and test/inference data (direct lookup).
    """
    global_mean = train_df[target_col].mean()

    district_stats = train_df.groupby('Main_District')[target_col].agg(['mean', 'count'])
    district_stats['smoothed_mean'] = (
        (district_stats['count'] * district_stats['mean'] + K_DISTRICT * global_mean) /
        (district_stats['count'] + K_DISTRICT)
    )
    district_map = district_stats['smoothed_mean'].to_dict()

    sub_stats = train_df.groupby('Official_Sub_Location')[target_col].agg(['mean', 'count'])
    sub_district_lookup = train_df.groupby('Official_Sub_Location')['Main_District'].first().to_dict()

    sub_map = {}
    for sub_loc, row in sub_stats.iterrows():
        district = sub_district_lookup.get(sub_loc)
        district_prior = district_map.get(district, global_mean)
        sub_map[sub_loc] = (row['count'] * row['mean'] + K_SUB * district_prior) / (row['count'] + K_SUB)

    return {
        'sub_map': sub_map,
        'district_map': district_map,
        'sub_to_district': sub_district_lookup,
        'global_mean': global_mean
    }


def apply_encoding(df, encoding):
    """Applies a fitted encoding map to any dataframe with Official_Sub_Location
    and Main_District columns. Falls back gracefully: sub-location -> district
    -> global mean, for locations not seen during fitting."""
    def encode_row(row):
        sub = row['Official_Sub_Location']
        if sub in encoding['sub_map']:
            return encoding['sub_map'][sub]
        district = row['Main_District']
        return encoding['district_map'].get(district, encoding['global_mean'])
    return df.apply(encode_row, axis=1)


def out_of_fold_encode_train(train_df, target_col='log_tct', n_folds=N_FOLDS, seed=42):
    """
    Encodes the TRAINING set using K-fold out-of-fold encoding: for each
    fold, the encoding is fit on the OTHER folds only, then applied to the
    held-out fold. This prevents a row's own price from leaking into its
    own encoded location value, which a naive single-pass encoding would do.
    """
    encoded = pd.Series(index=train_df.index, dtype=float)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold_train_idx, fold_holdout_idx in kf.split(train_df):
        fold_train = train_df.iloc[fold_train_idx]
        fold_holdout = train_df.iloc[fold_holdout_idx]
        fold_encoding = fit_hierarchical_encoding(fold_train, target_col)
        encoded.loc[fold_holdout.index] = apply_encoding(fold_holdout, fold_encoding)

    return encoded
# ============================================================
# [END ADDED]
# ============================================================


def extract_comprehensive_amenities(df, description_column):
    data = df.copy()
    data['clean_desc'] = data[description_column].fillna('').str.lower()

    comprehensive_features = {
        'has_prepaid_meter':     r'(pre.?paid|personal meter)',
        'has_screeded_walls':    r'(screed)',
        'has_fitted_kitchen':    r'(fitted kitchen|kitchen cab|cabs)',
        'has_clean_water':       r'(clean water|bore hole|borehole|treated|water board)',
        'has_steady_power':      r'(24.?7|24.?hr|constant light|power supply|inverter|gen|solar|diesel)',
        'has_ac':                r'(a/?c|air.?con)',
        'has_water_heater':      r'(water heater|hot water)',
        'has_internet':          r'(wifi|wi-fi|internet|fiber|fibre)',
        'is_ensuite':            r'(en.?suite|all rooms en)',
        'has_bq':                r'(bq|boys quarter|maid)',
        'has_pop_ceiling':       r'(pop|p\.o\.p)',
        'has_wardrobe':          r'(wardrobe|closet)',
        'has_balcony':           r'(balcony|terrace)',
        'is_secured':            r'(secur|guard|patrol|safe)',
        'in_gated_estate':       r'(gated|estate|call to enter)',
        'has_cctv':              r'(cctv|camera|surveillance)',
        'has_pool':              r'(pool|swimming)',
        'has_gym':               r'(gym|fitness)',
        'has_smart_home':        r'(smart|automated)',
        'has_elevator':          r'(elevator|lift)',
        'has_parking':           r'(park|garage)',
        'is_tarred':             r'(tarred|paved|interlock)',
        'has_easy_access':       r'(easy access|bus stop|accessible)',
        'prefers_working_class': r'(working class|corporate)',
        'has_jacuzzi':           r'(jacuzzi)',
        'is_premium_marketed':   r'(premium|luxury)',
        'is_urgent_listing':     r'(fastest finger|fcfs)',
    }

    for feature_name, pattern in comprehensive_features.items():
        data[feature_name] = data['clean_desc'].apply(
            lambda x: 1 if re.search(pattern, x) else 0
        )

    fenced_mask = data['clean_desc'].str.contains(r'fenced', regex=True)
    data['is_standalone_fenced'] = ((fenced_mask) & (data['in_gated_estate'] == 0)).astype(int)

    feature_names = list(comprehensive_features.keys()) + ['is_standalone_fenced']

    data = data.drop(columns=['clean_desc'])
    return data, feature_names


def engineer_features(input_csv):
    print("--- INITIATING FEATURE ENGINEERING (v2 - hierarchical target encoding) ---")

    df = pd.read_csv(input_csv)

    df['Property_Profile'] = (
        df['Raw_Description'].fillna('') + " " +
        df['Detailed_Attributes'].fillna('') + " " +
        df['Amenities'].fillna('')
    )

    print("Extracting verified Lagos real estate amenities via Regex...")
    df, extracted_feature_columns = extract_comprehensive_amenities(df, 'Property_Profile')
    print(f"Total amenity features extracted: {len(extracted_feature_columns)}")

    print("Extracting property sizing (Bedrooms, Bathrooms, Serviced/Furnished)...")
    df['Top_Icons'] = df['Top_Icons'].fillna('')
    df['Raw_Description'] = df['Raw_Description'].fillna('').str.lower()

    df['Bedrooms'] = df['Top_Icons'].str.extract(r'(\d+)\s*bedroom', flags=re.IGNORECASE)[0].fillna(0).astype(int)
    desc_beds = df['Raw_Description'].str.extract(r'(\d+)\s*-?\s*(?:bedroom|bed|brm|bdrm|bedeoom|beds)\b')[0].fillna(0).astype(int)
    df['Bedrooms'] = np.where(df['Bedrooms'] == 0, desc_beds, df['Bedrooms'])

    implied_1_bed = df['Raw_Description'].str.contains(
        r'(?:mini\s*flat|self\s*con|room\s*and\s*parlour|room\s*&\s*parlour|selfcon|bq|room\s*self|roomself|single\s*room)',
        na=False
    )
    df['Bedrooms'] = np.where((df['Bedrooms'] == 0) & implied_1_bed, 1, df['Bedrooms'])

    df['Bathrooms'] = df['Top_Icons'].str.extract(r'(\d+)\s*bathroom', flags=re.IGNORECASE)[0].fillna(0).astype(int)
    desc_baths = df['Raw_Description'].str.extract(r'(\d+)\s*(?:bath|bathroom|bathrooms|baths|toilets)\b')[0].fillna(0).astype(int)
    df['Bathrooms'] = np.where(df['Bathrooms'] == 0, desc_baths, df['Bathrooms'])

    df['Is_Serviced'] = df['Is_Serviced'].fillna(0).astype(int)
    df['Is_Furnished'] = df['Is_Furnished'].fillna(0).astype(int)

    structural_features = ['Bedrooms', 'Bathrooms', 'Is_Serviced', 'Is_Furnished']

    print("Applying np.log1p transformation to Target Variable (TCT)...")
    df['log_tct'] = np.log1p(df['TCT'])

    # ============================================================
    # [MODIFIED] Split FIRST, before any location encoding, so the
    # encoding can never see test data. This replaces the old
    # OneHotEncoder + ColumnTransformer + Consolidated_Location logic.
    # ============================================================
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)
    train_df = df.loc[train_idx].copy()
    test_df = df.loc[test_idx].copy()
    print(f"Dataset Split: {len(train_df)} Training rows | {len(test_df)} Testing rows")

    print(f"Fitting hierarchical target encoding on TRAINING data only "
          f"(K_SUB={K_SUB}, K_DISTRICT={K_DISTRICT})...")
    full_train_encoding = fit_hierarchical_encoding(train_df, target_col='log_tct')

    print(f"Encoding training set via {N_FOLDS}-fold out-of-fold (leakage-safe)...")
    train_df['location_encoded'] = out_of_fold_encode_train(train_df, target_col='log_tct')

    print("Encoding test set using training-only fitted map...")
    test_df['location_encoded'] = apply_encoding(test_df, full_train_encoding)

    print(f"Sub-locations with their own encoding: {len(full_train_encoding['sub_map'])} "
          f"(out of {df['Official_Sub_Location'].nunique()} total in dataset)")
    print(f"Districts covered: {len(full_train_encoding['district_map'])}")
    # ============================================================
    # [END MODIFIED]
    # ============================================================

    feature_cols = ['location_encoded'] + extracted_feature_columns + structural_features

    X_train = train_df[feature_cols]
    y_train = train_df['log_tct']
    X_test = test_df[feature_cols]
    y_test = test_df['log_tct']

    print(f"\nFinal feature dimensions: {X_train.shape[1]} (dense, vs 162 for OHE)")

    # [MODIFIED] Dense output now — plain CSV instead of scipy sparse .npz,
    # since target encoding produces continuous values, not sparse dummies.
    print("Saving dense feature matrices...")
    X_train.to_csv('X_train_v2.csv', index=False)
    X_test.to_csv('X_test_v2.csv', index=False)
    y_train.to_csv('y_train_v2.csv', index=False)
    y_test.to_csv('y_test_v2.csv', index=False)

    # [ADDED] Saves the full encoding map (sub-location, district, and global
    # fallback layers) fit on ALL training data. This is what future
    # inference code needs to encode a brand-new property's location,
    # including sub-locations from your dictionary that were never scraped
    # at all — they'll gracefully fall back to their district's value.
    joblib.dump(full_train_encoding, 'location_encoding_map_v2.pkl')

    print("Success. Files: X_train_v2.csv, X_test_v2.csv,")
    print("                y_train_v2.csv, y_test_v2.csv, location_encoding_map_v2.pkl")


if __name__ == "__main__":
    engineer_features('lagos_rent_spatially_capped_v2.csv')







































# # feature_engineering_pipeline_v2.py -- Take 2
# # [MODIFIED] Added new amenity features discovered via keyword_discovery.py,
# # validated against the current best dataset (lagos_rent_spatially_capped_v2.csv,
# # the one that produced R2 0.7792):
# #   - has_jacuzzi: new luxury amenity signal, not previously captured
# #   - is_premium_marketed: marketing language ("premium"/"luxury") correlating
# #     with genuinely higher-priced listings
# #   - is_urgent_listing: "fastest finger first" / FCFS listings correlate with
# #     lower prices (budget/urgent listings) — a real, non-obvious negative signal
# #   - is_standalone_fenced: a standalone fenced compound is NOT the same as a
# #     gated estate, and actually correlates with LOWER price. Previously
# #     'fenced' language wasn't captured at all, meaning this distinction was
# #     invisible to the model. Depends on in_gated_estate already having been
# #     computed, so it's only 1 when 'fenced' appears WITHOUT also being in a
# #     gated estate.
# #   - has_steady_power: extended to also catch 'diesel' (as in "diesel
# #     generator"), which was previously invisible to this feature.
# #
# # NOTE: the two "bedroom regex" fixes discussed earlier were found, on
# # verification, to already work correctly in the existing patterns. No change
# # was needed there — flagging this so it's clear nothing was silently altered.
# #
# # TO REVERT: Delete the has_jacuzzi, is_premium_marketed, and is_urgent_listing
# # lines from comprehensive_features. Remove 'diesel' from has_steady_power's
# # pattern. Delete the is_standalone_fenced post-processing block.

# import pandas as pd
# import numpy as np
# import scipy.sparse
# import re
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# import joblib


# def extract_comprehensive_amenities(df, description_column):
#     data = df.copy()
#     data['clean_desc'] = data[description_column].fillna('').str.lower()

#     comprehensive_features = {
#         'has_prepaid_meter':     r'(pre.?paid|personal meter)',
#         'has_screeded_walls':    r'(screed)',
#         'has_fitted_kitchen':    r'(fitted kitchen|kitchen cab|cabs)',
#         'has_clean_water':       r'(clean water|bore hole|borehole|treated|water board)',
#         # [MODIFIED] Added 'diesel' — previously only caught 'gen'/'inverter'/'solar',
#         # missing "diesel generator" phrasing entirely.
#         'has_steady_power':      r'(24.?7|24.?hr|constant light|power supply|inverter|gen|solar|diesel)',
#         'has_ac':                r'(a/?c|air.?con)',
#         'has_water_heater':      r'(water heater|hot water)',
#         'has_internet':          r'(wifi|wi-fi|internet|fiber|fibre)',
#         'is_ensuite':            r'(en.?suite|all rooms en)',
#         'has_bq':                r'(bq|boys quarter|maid)',
#         'has_pop_ceiling':       r'(pop|p\.o\.p)',
#         'has_wardrobe':          r'(wardrobe|closet)',
#         'has_balcony':           r'(balcony|terrace)',
#         'is_secured':            r'(secur|guard|patrol|safe)',
#         'in_gated_estate':       r'(gated|estate|call to enter)',
#         'has_cctv':              r'(cctv|camera|surveillance)',
#         'has_pool':              r'(pool|swimming)',
#         'has_gym':               r'(gym|fitness)',
#         'has_smart_home':        r'(smart|automated)',
#         'has_elevator':          r'(elevator|lift)',
#         'has_parking':           r'(park|garage)',
#         'is_tarred':             r'(tarred|paved|interlock)',
#         'has_easy_access':       r'(easy access|bus stop|accessible)',
#         'prefers_working_class': r'(working class|corporate)',
#         # ============================================================
#         # [ADDED] New features found via keyword_discovery.py, validated
#         # against actual price effect and statistical significance on the
#         # current best dataset.
#         # ============================================================
#         'has_jacuzzi':           r'(jacuzzi)',
#         'is_premium_marketed':   r'(premium|luxury)',
#         'is_urgent_listing':     r'(fastest finger|fcfs)',
#         # ============================================================
#         # [END ADDED]
#         # ============================================================
#     }

#     for feature_name, pattern in comprehensive_features.items():
#         data[feature_name] = data['clean_desc'].apply(
#             lambda x: 1 if re.search(pattern, x) else 0
#         )

#     # ============================================================
#     # [ADDED] is_standalone_fenced — depends on in_gated_estate already
#     # being computed above. A standalone fenced compound is NOT the same
#     # as a gated estate and correlates with LOWER price (-68% effect,
#     # p < 0.001 on the discovery run). Only fires when 'fenced' language
#     # is present but the listing is NOT already flagged as a gated estate,
#     # so the two features don't overlap and dilute each other.
#     #
#     # TO REVERT: delete this block and remove 'is_standalone_fenced' from
#     # the returned feature list below.
#     # ============================================================
#     fenced_mask = data['clean_desc'].str.contains(r'fenced', regex=True)
#     data['is_standalone_fenced'] = ((fenced_mask) & (data['in_gated_estate'] == 0)).astype(int)
#     # ============================================================
#     # [END ADDED]
#     # ============================================================

#     feature_names = list(comprehensive_features.keys()) + ['is_standalone_fenced']

#     data = data.drop(columns=['clean_desc'])
#     return data, feature_names


# RARE_THRESHOLD = 5

# DISTRICT_MAP = {
#     'Lekki':    ['Lekki', 'Ikate', 'Agungi', 'Chevron', 'Osapa', 'Ikota', 'Marwa', 'Igbo Efon'],
#     'Ajah':     ['Ajah', 'Sangotedo', 'Badore', 'Awoyaya', 'Bogije', 'Abraham Adesanya',
#                  'Ibeju', 'Eleko', 'Epe', 'Ado'],
#     'VI':       ['Victoria Island', 'Adeola', 'Akin Adesola', 'Ahmadu Bello', 'Ozumba'],
#     'Ikoyi':    ['Ikoyi', 'Osborne', 'Parkview', 'Old Ikoyi', 'Banana Island'],
#     'Ikeja':    ['Ikeja', 'Allen', 'Alausa', 'Oregun', 'Adeniyi Jones', 'Toyin',
#                  'Acme', 'Ogba', 'Agidingbi'],
#     'Maryland': ['Maryland', 'Mende', 'Ilupeju'],
#     'Ojodu':    ['Ojodu', 'Berger', 'Isheri', 'Magodo', 'Shangisha', 'Omole', 'Ketu', 'Ojota'],
#     'Gbagada':  ['Gbagada', 'Soluyi', 'Ifako', 'Pedro', 'Ogudu'],
#     'Yaba':     ['Yaba', 'Alagomeji', 'Akoka', 'Sabo', 'Abule Ijesha', 'Bariga', 'Shomolu'],
#     'Surulere': ['Surulere', 'Ojuelegba', 'Aguda', 'Itire', 'Bode Thomas', 'Eric Moore'],
#     'Isolo':    ['Isolo', 'Ajao Estate', 'Mafoluku', 'Oshodi', 'Ejigbo'],
#     'Ikorodu':  ['Ikorodu', 'Ajuwon', 'Akute', 'Agric', 'Imota', 'Kosofe', 'Mile 12'],
#     'Alimosho': ['Alimosho', 'Ipaja', 'Egbeda', 'Akesan', 'Abule Egba', 'Meiran', 'Agege'],
#     'Okota':    ['Okota', 'Ago Palace'],
#     'Amuwo':    ['Amuwo', 'Festac', 'Mile 2', 'Apapa', 'Kirikiri'],
#     'Badagry':  ['Badagry', 'Okokomaiko', 'Iyana Iba'],
# }


# def get_consolidated_location(loc_string, loc_counts_dict):
#     if loc_counts_dict.get(loc_string, 0) >= RARE_THRESHOLD:
#         return loc_string
#     loc_lower = loc_string.lower()
#     for district, keywords in DISTRICT_MAP.items():
#         if any(kw.lower() in loc_lower for kw in keywords):
#             return district
#     return 'Lagos Other'


# def engineer_features(input_csv):
#     print("--- INITIATING FEATURE ENGINEERING (v2 - expanded amenities) ---")

#     df = pd.read_csv(input_csv)

#     df['Property_Profile'] = (
#         df['Raw_Description'].fillna('') + " " +
#         df['Detailed_Attributes'].fillna('') + " " +
#         df['Amenities'].fillna('')
#     )

#     print("Extracting verified Lagos real estate amenities via Regex...")
#     df, extracted_feature_columns = extract_comprehensive_amenities(df, 'Property_Profile')
#     print(f"Total amenity features extracted: {len(extracted_feature_columns)}")

#     print("Extracting property sizing (Bedrooms, Bathrooms, Serviced/Furnished)...")
#     df['Top_Icons'] = df['Top_Icons'].fillna('')
#     df['Raw_Description'] = df['Raw_Description'].fillna('').str.lower()

#     df['Bedrooms'] = df['Top_Icons'].str.extract(r'(\d+)\s*bedroom', flags=re.IGNORECASE)[0].fillna(0).astype(int)
#     desc_beds = df['Raw_Description'].str.extract(r'(\d+)\s*-?\s*(?:bedroom|bed|brm|bdrm|bedeoom|beds)\b')[0].fillna(0).astype(int)
#     df['Bedrooms'] = np.where(df['Bedrooms'] == 0, desc_beds, df['Bedrooms'])

#     implied_1_bed = df['Raw_Description'].str.contains(
#         r'(?:mini\s*flat|self\s*con|room\s*and\s*parlour|room\s*&\s*parlour|selfcon|bq|room\s*self|roomself|single\s*room)',
#         na=False
#     )
#     df['Bedrooms'] = np.where((df['Bedrooms'] == 0) & implied_1_bed, 1, df['Bedrooms'])

#     df['Bathrooms'] = df['Top_Icons'].str.extract(r'(\d+)\s*bathroom', flags=re.IGNORECASE)[0].fillna(0).astype(int)
#     desc_baths = df['Raw_Description'].str.extract(r'(\d+)\s*(?:bath|bathroom|bathrooms|baths|toilets)\b')[0].fillna(0).astype(int)
#     df['Bathrooms'] = np.where(df['Bathrooms'] == 0, desc_baths, df['Bathrooms'])

#     df['Is_Serviced'] = df['Is_Serviced'].fillna(0).astype(int)
#     df['Is_Furnished'] = df['Is_Furnished'].fillna(0).astype(int)

#     structural_features = ['Bedrooms', 'Bathrooms', 'Is_Serviced', 'Is_Furnished']

#     print(f"Consolidating rare sub-locations (threshold = {RARE_THRESHOLD} rows)...")
#     loc_counts_dict = df['Cleaned_Location'].value_counts().to_dict()
#     df['Consolidated_Location'] = df['Cleaned_Location'].apply(
#         lambda x: get_consolidated_location(x, loc_counts_dict)
#     )
#     unique_before = df['Cleaned_Location'].nunique()
#     unique_after = df['Consolidated_Location'].nunique()
#     print(f"  Location labels: {unique_before} -> {unique_after} unique values")

#     features_to_keep = ['Consolidated_Location'] + extracted_feature_columns + structural_features
#     X = df[features_to_keep]

#     print("Applying np.log1p transformation to Target Variable (TCT)...")
#     y = np.log1p(df['TCT'])

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     print(f"Dataset Split: {len(X_train)} Training rows | {len(X_test)} Testing rows")

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('location', OneHotEncoder(handle_unknown='ignore'), ['Consolidated_Location'])
#         ],
#         remainder='passthrough'
#     )

#     print("Fitting OneHotEncoder on consolidated location labels...")
#     X_train_matrix = preprocessor.fit_transform(X_train)
#     X_test_matrix = preprocessor.transform(X_test)

#     print(f"Extraction complete. Feature dimensions: {X_train_matrix.shape[1]}")

#     print("\nSaving sparse matrices...")
#     scipy.sparse.save_npz('X_train_v2.npz', X_train_matrix)
#     scipy.sparse.save_npz('X_test_v2.npz', X_test_matrix)

#     y_train.to_csv('y_train_v2.csv', index=False)
#     y_test.to_csv('y_test_v2.csv', index=False)

#     joblib.dump(preprocessor, 'feature_preprocessor_v2.pkl')

#     print("Success. Files: X_train_v2.npz, X_test_v2.npz,")
#     print("                y_train_v2.csv, y_test_v2.csv, feature_preprocessor_v2.pkl")


# if __name__ == "__main__":
#     engineer_features('lagos_rent_spatially_capped_v2.csv')



































# # feature_engineering_pipeline_v2.py -- Take 1
# # Identical logic to feature_engineering_pipeline_sparse.py (the version
# # using OneHotEncoder + Consolidated_Location). Only the input/output filenames change, so it runs against the v2.

# import pandas as pd
# import numpy as np
# import scipy.sparse
# import re
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# import joblib


# def extract_comprehensive_amenities(df, description_column):
#     data = df.copy()
#     data['clean_desc'] = data[description_column].fillna('').str.lower()

#     comprehensive_features = {
#         'has_prepaid_meter':     r'(pre.?paid|personal meter)',
#         'has_screeded_walls':    r'(screed)',
#         'has_fitted_kitchen':    r'(fitted kitchen|kitchen cab|cabs)',
#         'has_clean_water':       r'(clean water|bore hole|borehole|treated|water board)',
#         'has_steady_power':      r'(24.?7|24.?hr|constant light|power supply|inverter|gen|solar)',
#         'has_ac':                r'(a/?c|air.?con)',
#         'has_water_heater':      r'(water heater|hot water)',
#         'has_internet':          r'(wifi|wi-fi|internet|fiber|fibre)',
#         'is_ensuite':            r'(en.?suite|all rooms en)',
#         'has_bq':                r'(bq|boys quarter|maid)',
#         'has_pop_ceiling':       r'(pop|p\.o\.p)',
#         'has_wardrobe':          r'(wardrobe|closet)',
#         'has_balcony':           r'(balcony|terrace)',
#         'is_secured':            r'(secur|guard|patrol|safe)',
#         'in_gated_estate':       r'(gated|estate|call to enter)',
#         'has_cctv':              r'(cctv|camera|surveillance)',
#         'has_pool':              r'(pool|swimming)',
#         'has_gym':               r'(gym|fitness)',
#         'has_smart_home':        r'(smart|automated)',
#         'has_elevator':          r'(elevator|lift)',
#         'has_parking':           r'(park|garage)',
#         'is_tarred':             r'(tarred|paved|interlock)',
#         'has_easy_access':       r'(easy access|bus stop|accessible)',
#         'prefers_working_class': r'(working class|corporate)'
#     }

#     for feature_name, pattern in comprehensive_features.items():
#         data[feature_name] = data['clean_desc'].apply(
#             lambda x: 1 if re.search(pattern, x) else 0
#         )

#     data = data.drop(columns=['clean_desc'])
#     return data, list(comprehensive_features.keys())


# RARE_THRESHOLD = 5

# DISTRICT_MAP = {
#     'Lekki':    ['Lekki', 'Ikate', 'Agungi', 'Chevron', 'Osapa', 'Ikota', 'Marwa', 'Igbo Efon'],
#     'Ajah':     ['Ajah', 'Sangotedo', 'Badore', 'Awoyaya', 'Bogije', 'Abraham Adesanya',
#                  'Ibeju', 'Eleko', 'Epe', 'Ado'],
#     'VI':       ['Victoria Island', 'Adeola', 'Akin Adesola', 'Ahmadu Bello', 'Ozumba'],
#     'Ikoyi':    ['Ikoyi', 'Osborne', 'Parkview', 'Old Ikoyi', 'Banana Island'],
#     'Ikeja':    ['Ikeja', 'Allen', 'Alausa', 'Oregun', 'Adeniyi Jones', 'Toyin',
#                  'Acme', 'Ogba', 'Agidingbi'],
#     'Maryland': ['Maryland', 'Mende', 'Ilupeju'],
#     'Ojodu':    ['Ojodu', 'Berger', 'Isheri', 'Magodo', 'Shangisha', 'Omole', 'Ketu', 'Ojota'],
#     'Gbagada':  ['Gbagada', 'Soluyi', 'Ifako', 'Pedro', 'Ogudu'],
#     'Yaba':     ['Yaba', 'Alagomeji', 'Akoka', 'Sabo', 'Abule Ijesha', 'Bariga', 'Shomolu'],
#     'Surulere': ['Surulere', 'Ojuelegba', 'Aguda', 'Itire', 'Bode Thomas', 'Eric Moore'],
#     'Isolo':    ['Isolo', 'Ajao Estate', 'Mafoluku', 'Oshodi', 'Ejigbo'],
#     'Ikorodu':  ['Ikorodu', 'Ajuwon', 'Akute', 'Agric', 'Imota', 'Kosofe', 'Mile 12'],
#     'Alimosho': ['Alimosho', 'Ipaja', 'Egbeda', 'Akesan', 'Abule Egba', 'Meiran', 'Agege'],
#     'Okota':    ['Okota', 'Ago Palace'],
#     'Amuwo':    ['Amuwo', 'Festac', 'Mile 2', 'Apapa', 'Kirikiri'],
#     'Badagry':  ['Badagry', 'Okokomaiko', 'Iyana Iba'],
# }


# def get_consolidated_location(loc_string, loc_counts_dict):
#     if loc_counts_dict.get(loc_string, 0) >= RARE_THRESHOLD:
#         return loc_string
#     loc_lower = loc_string.lower()
#     for district, keywords in DISTRICT_MAP.items():
#         if any(kw.lower() in loc_lower for kw in keywords):
#             return district
#     return 'Lagos Other'


# def engineer_features(input_csv):
#     print("--- INITIATING FEATURE ENGINEERING (v2 - reordered pipeline) ---")

#     df = pd.read_csv(input_csv)

#     df['Property_Profile'] = (
#         df['Raw_Description'].fillna('') + " " +
#         df['Detailed_Attributes'].fillna('') + " " +
#         df['Amenities'].fillna('')
#     )

#     print("Extracting verified Lagos real estate amenities via Regex...")
#     df, extracted_feature_columns = extract_comprehensive_amenities(df, 'Property_Profile')

#     print("Extracting property sizing (Bedrooms, Bathrooms, Serviced/Furnished)...")
#     df['Top_Icons'] = df['Top_Icons'].fillna('')
#     df['Raw_Description'] = df['Raw_Description'].fillna('').str.lower()

#     df['Bedrooms'] = df['Top_Icons'].str.extract(r'(\d+)\s*bedroom', flags=re.IGNORECASE)[0].fillna(0).astype(int)
#     desc_beds = df['Raw_Description'].str.extract(r'(\d+)\s*-?\s*(?:bedroom|bed|brm|bdrm|bedeoom|beds)\b')[0].fillna(0).astype(int)
#     df['Bedrooms'] = np.where(df['Bedrooms'] == 0, desc_beds, df['Bedrooms'])

#     implied_1_bed = df['Raw_Description'].str.contains(
#         r'(?:mini\s*flat|self\s*con|room\s*and\s*parlour|room\s*&\s*parlour|selfcon|bq|room\s*self|roomself|single\s*room)',
#         na=False
#     )
#     df['Bedrooms'] = np.where((df['Bedrooms'] == 0) & implied_1_bed, 1, df['Bedrooms'])

#     df['Bathrooms'] = df['Top_Icons'].str.extract(r'(\d+)\s*bathroom', flags=re.IGNORECASE)[0].fillna(0).astype(int)
#     desc_baths = df['Raw_Description'].str.extract(r'(\d+)\s*(?:bath|bathroom|bathrooms|baths|toilets)\b')[0].fillna(0).astype(int)
#     df['Bathrooms'] = np.where(df['Bathrooms'] == 0, desc_baths, df['Bathrooms'])

#     df['Is_Serviced'] = df['Is_Serviced'].fillna(0).astype(int)
#     df['Is_Furnished'] = df['Is_Furnished'].fillna(0).astype(int)

#     structural_features = ['Bedrooms', 'Bathrooms', 'Is_Serviced', 'Is_Furnished']

#     print(f"Consolidating rare sub-locations (threshold = {RARE_THRESHOLD} rows)...")
#     loc_counts_dict = df['Cleaned_Location'].value_counts().to_dict()
#     df['Consolidated_Location'] = df['Cleaned_Location'].apply(
#         lambda x: get_consolidated_location(x, loc_counts_dict)
#     )
#     unique_before = df['Cleaned_Location'].nunique()
#     unique_after = df['Consolidated_Location'].nunique()
#     print(f"  Location labels: {unique_before} -> {unique_after} unique values")

#     features_to_keep = ['Consolidated_Location'] + extracted_feature_columns + structural_features
#     X = df[features_to_keep]

#     print("Applying np.log1p transformation to Target Variable (TCT)...")
#     y = np.log1p(df['TCT'])

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     print(f"Dataset Split: {len(X_train)} Training rows | {len(X_test)} Testing rows")

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('location', OneHotEncoder(handle_unknown='ignore'), ['Consolidated_Location'])
#         ],
#         remainder='passthrough'
#     )

#     print("Fitting OneHotEncoder on consolidated location labels...")
#     X_train_matrix = preprocessor.fit_transform(X_train)
#     X_test_matrix = preprocessor.transform(X_test)

#     print(f"Extraction complete. Feature dimensions: {X_train_matrix.shape[1]}")

#     # [MODIFIED] All output filenames suffixed with _v2 so nothing overwrites
#     # your live model's files.
#     print("\nSaving sparse matrices...")
#     scipy.sparse.save_npz('X_train_v2.npz', X_train_matrix)
#     scipy.sparse.save_npz('X_test_v2.npz', X_test_matrix)

#     y_train.to_csv('y_train_v2.csv', index=False)
#     y_test.to_csv('y_test_v2.csv', index=False)

#     joblib.dump(preprocessor, 'feature_preprocessor_v2.pkl')

#     print("Success. Files: X_train_v2.npz, X_test_v2.npz,")
#     print("                y_train_v2.csv, y_test_v2.csv, feature_preprocessor_v2.pkl")


# if __name__ == "__main__":
#     engineer_features('lagos_rent_spatially_capped_v2.csv')