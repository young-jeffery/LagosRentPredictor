# feature_engineering_pipeline_v3.py -- Take 1
# [MODIFIED FROM v2] Adds Property_Type as a new categorical feature,
# classified via property_type_classifier.py and target-encoded the same
# principled way as location: fit on TRAIN ONLY, out-of-fold for train,
# direct lookup for test/inference, with graceful fallback to the global
# mean for any category not seen during fitting (shouldn't happen with
# only 12 fixed categories and ~2900 rows, but kept for safety symmetry
# with the location encoder).
#
# Only single-level smoothing is used here (no hierarchy) since property
# type has no natural parent category the way sub-location has district.
#
# Verified before wiring in: all 12 categories statistically significant
# (p<0.01, worst case Maisonette n=10 p=3.3e-4), effect sizes up to +330%
# vs. overall median (Detached) down to -66% (Studio). See chat history
# for the validation run.
#
# NOTE: also fixed a substring-priority bug from the original proposal
# (Detached was ordered before Semi-detached; since "semi-detached"
# contains "detached", semi-detached listings would have been silently
# swallowed into the Detached bucket). property_type_classifier.py has
# the corrected order -- see comments there.
#
# Everything else is identical to feature_engineering_pipeline_v2.py.
#
# TO REVERT: use feature_engineering_pipeline_v2.py -- untouched, still
# produces your current production X_train_v2.csv / xgb_tuned_model_v3.pkl
# (monotonic constraint) inputs.

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, KFold
import joblib

from property_type_classifier import add_property_type_column, ALL_PROPERTY_TYPES

K_SUB = 10
K_DISTRICT = 30
K_PROPERTY_TYPE = 15   # [NEW] smoothing strength for property type -> global mean.
                       # Similar order of magnitude to K_SUB since category
                       # sample sizes span a similar range (10 to ~1500).
N_FOLDS = 5


def fit_hierarchical_encoding(train_df, target_col='log_tct'):
    """Unchanged from v2 -- fits sub-location and district encoding maps
    using TRAIN DATA ONLY."""
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
    """Unchanged from v2."""
    def encode_row(row):
        sub = row['Official_Sub_Location']
        if sub in encoding['sub_map']:
            return encoding['sub_map'][sub]
        district = row['Main_District']
        return encoding['district_map'].get(district, encoding['global_mean'])
    return df.apply(encode_row, axis=1)


def out_of_fold_encode_train(train_df, target_col='log_tct', n_folds=N_FOLDS, seed=42):
    """Unchanged from v2."""
    encoded = pd.Series(index=train_df.index, dtype=float)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold_train_idx, fold_holdout_idx in kf.split(train_df):
        fold_train = train_df.iloc[fold_train_idx]
        fold_holdout = train_df.iloc[fold_holdout_idx]
        fold_encoding = fit_hierarchical_encoding(fold_train, target_col)
        encoded.loc[fold_holdout.index] = apply_encoding(fold_holdout, fold_encoding)

    return encoded


# ============================================================
# [NEW] Single-level smoothed target encoding for Property_Type.
# Same math as the district level above, just no sub-level underneath it.
# ============================================================
def fit_property_type_encoding(train_df, target_col='log_tct'):
    global_mean = train_df[target_col].mean()
    type_stats = train_df.groupby('Property_Type')[target_col].agg(['mean', 'count'])
    type_stats['smoothed_mean'] = (
        (type_stats['count'] * type_stats['mean'] + K_PROPERTY_TYPE * global_mean) /
        (type_stats['count'] + K_PROPERTY_TYPE)
    )
    return {
        'type_map': type_stats['smoothed_mean'].to_dict(),
        'global_mean': global_mean
    }


def apply_property_type_encoding(df, encoding):
    return df['Property_Type'].map(encoding['type_map']).fillna(encoding['global_mean'])


def out_of_fold_encode_property_type(train_df, target_col='log_tct', n_folds=N_FOLDS, seed=42):
    encoded = pd.Series(index=train_df.index, dtype=float)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold_train_idx, fold_holdout_idx in kf.split(train_df):
        fold_train = train_df.iloc[fold_train_idx]
        fold_holdout = train_df.iloc[fold_holdout_idx]
        fold_encoding = fit_property_type_encoding(fold_train, target_col)
        encoded.loc[fold_holdout.index] = apply_property_type_encoding(fold_holdout, fold_encoding)

    return encoded
# ============================================================
# [END NEW]
# ============================================================


def extract_comprehensive_amenities(df, description_column):
    """Unchanged from v2."""
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
    print("--- INITIATING FEATURE ENGINEERING (v3 - + Property_Type) ---")

    df = pd.read_csv(input_csv)

    df['Property_Profile'] = (
        df['Raw_Description'].fillna('') + " " +
        df['Detailed_Attributes'].fillna('') + " " +
        df['Amenities'].fillna('')
    )

    print("Extracting verified Lagos real estate amenities via Regex...")
    df, extracted_feature_columns = extract_comprehensive_amenities(df, 'Property_Profile')
    print(f"Total amenity features extracted: {len(extracted_feature_columns)}")

    # [NEW] Property_Type classification. Uses Top_Icons in addition to the
    # usual text fields -- structural type is sometimes stated there even
    # when absent from free text.
    print("Classifying Property_Type (Penthouse/Detached/.../Flat-Apartment)...")
    df['Property_Type'] = add_property_type_column(
        df, ['Raw_Description', 'Detailed_Attributes', 'Amenities', 'Top_Icons']
    )
    print("Property_Type distribution:")
    print(df['Property_Type'].value_counts())

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

    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)
    train_df = df.loc[train_idx].copy()
    test_df = df.loc[test_idx].copy()
    print(f"Dataset Split: {len(train_df)} Training rows | {len(test_df)} Testing rows")

    print(f"Fitting hierarchical location target encoding on TRAINING data only "
          f"(K_SUB={K_SUB}, K_DISTRICT={K_DISTRICT})...")
    full_train_encoding = fit_hierarchical_encoding(train_df, target_col='log_tct')

    print(f"Encoding training set location via {N_FOLDS}-fold out-of-fold (leakage-safe)...")
    train_df['location_encoded'] = out_of_fold_encode_train(train_df, target_col='log_tct')

    print("Encoding test set location using training-only fitted map...")
    test_df['location_encoded'] = apply_encoding(test_df, full_train_encoding)

    print(f"Sub-locations with their own encoding: {len(full_train_encoding['sub_map'])} "
          f"(out of {df['Official_Sub_Location'].nunique()} total in dataset)")
    print(f"Districts covered: {len(full_train_encoding['district_map'])}")

    # [NEW] Property_Type target encoding -- same leakage-safe pattern.
    print(f"\nFitting Property_Type target encoding on TRAINING data only "
          f"(K_PROPERTY_TYPE={K_PROPERTY_TYPE})...")
    full_property_type_encoding = fit_property_type_encoding(train_df, target_col='log_tct')

    print(f"Encoding training set Property_Type via {N_FOLDS}-fold out-of-fold (leakage-safe)...")
    train_df['property_type_encoded'] = out_of_fold_encode_property_type(train_df, target_col='log_tct')

    print("Encoding test set Property_Type using training-only fitted map...")
    test_df['property_type_encoded'] = apply_property_type_encoding(test_df, full_property_type_encoding)

    print(f"Property types with their own encoding: {len(full_property_type_encoding['type_map'])} "
          f"(out of {len(ALL_PROPERTY_TYPES)} possible categories)")

    # [MODIFIED] property_type_encoded inserted right after location_encoded --
    # both are target-encoded continuous location/category signals, grouped
    # together at the front of the feature vector.
    feature_cols = ['location_encoded', 'property_type_encoded'] + extracted_feature_columns + structural_features

    X_train = train_df[feature_cols]
    y_train = train_df['log_tct']
    X_test = test_df[feature_cols]
    y_test = test_df['log_tct']

    print(f"\nFinal feature dimensions: {X_train.shape[1]} (v2 had 33, +1 for property_type_encoded)")

    print("Saving dense feature matrices...")
    X_train.to_csv('X_train_v3.csv', index=False)
    X_test.to_csv('X_test_v3.csv', index=False)
    y_train.to_csv('y_train_v3.csv', index=False)
    y_test.to_csv('y_test_v3.csv', index=False)

    joblib.dump(full_train_encoding, 'location_encoding_map_v3.pkl')
    joblib.dump(full_property_type_encoding, 'property_type_encoding_map_v3.pkl')

    print("Success. Files: X_train_v3.csv, X_test_v3.csv,")
    print("                y_train_v3.csv, y_test_v3.csv,")
    print("                location_encoding_map_v3.pkl, property_type_encoding_map_v3.pkl")


if __name__ == "__main__":
    engineer_features('lagos_rent_spatially_capped_v2.csv')