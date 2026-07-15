# Fifth attempt at the feature engineering pipeline.
#
# Previous attempts:
#   1st — TF-IDF on raw text (too noisy, included agent spam)
#   2nd — Domain-expert regex amenities, OHE on location
#   3rd — Same as 2nd + structural features (Bedrooms, Bathrooms) [was best baseline]
#   4th — Added Isolation Forest + switched to TargetEncoder for location
#          Result: R2 0.6446, MAPE 82.78% — WORSE than 3rd (0.6949, 76.42%)
#          Root causes: TargetEncoder learned corrupt weights from commercial
#          listings still in the data; Isolation Forest dropped valid properties.
#
# [MODIFIED] This attempt returns to OneHotEncoder (like the 3rd) but adds a
# Location Consolidation step to solve the sparsity problem that was making OHE
# poor. The dataset had 944 unique location labels across 2,772 rows — 662
# appeared in only 1 row each, producing OHE columns the model cannot learn.
# Consolidating sub-locations with < 5 rows to their parent district cuts
# unique labels from ~940 to ~141, making each OHE column dense and learnable.

import pandas as pd
import numpy as np
import scipy.sparse
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# [REMOVED] These were used by the 4th attempt. Restore if switching back.
# from sklearn.ensemble import IsolationForest
# from category_encoders import TargetEncoder


def extract_comprehensive_amenities(df, description_column):
    # Unchanged from 3rd/4th attempt
    data = df.copy()
    data['clean_desc'] = data[description_column].fillna('').str.lower()

    comprehensive_features = {
        'has_prepaid_meter':     r'(pre.?paid|personal meter)',
        'has_screeded_walls':    r'(screed)',
        'has_fitted_kitchen':    r'(fitted kitchen|kitchen cab|cabs)',
        'has_clean_water':       r'(clean water|bore hole|borehole|treated|water board)',
        'has_steady_power':      r'(24.?7|24.?hr|constant light|power supply|inverter|gen|solar)',
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
        'prefers_working_class': r'(working class|corporate)'
    }

    for feature_name, pattern in comprehensive_features.items():
        data[feature_name] = data['clean_desc'].apply(
            lambda x: 1 if re.search(pattern, x) else 0
        )

    data = data.drop(columns=['clean_desc'])
    return data, list(comprehensive_features.keys())


# ============================================================
# [ADDED] LOCATION CONSOLIDATION LOGIC
#
# Maps known Lagos sub-locations to their parent district.
# Sub-locations with fewer than RARE_THRESHOLD rows get replaced
# with the district name so OHE has dense, learnable columns.
#
# TO REVERT: Delete DISTRICT_MAP, RARE_THRESHOLD, and
# get_consolidated_location. In engineer_features, delete the
# consolidation block and change every 'Consolidated_Location'
# back to 'Cleaned_Location'.
# ============================================================
RARE_THRESHOLD = 5

DISTRICT_MAP = {
    'Lekki':    ['Lekki', 'Ikate', 'Agungi', 'Chevron', 'Osapa', 'Ikota', 'Marwa', 'Igbo Efon'],
    'Ajah':     ['Ajah', 'Sangotedo', 'Badore', 'Awoyaya', 'Bogije', 'Abraham Adesanya',
                 'Ibeju', 'Eleko', 'Epe', 'Ado'],
    'VI':       ['Victoria Island', 'Adeola', 'Akin Adesola', 'Ahmadu Bello', 'Ozumba'],
    'Ikoyi':    ['Ikoyi', 'Osborne', 'Parkview', 'Old Ikoyi', 'Banana Island'],
    'Ikeja':    ['Ikeja', 'Allen', 'Alausa', 'Oregun', 'Adeniyi Jones', 'Toyin',
                 'Acme', 'Ogba', 'Agidingbi'],
    'Maryland': ['Maryland', 'Mende', 'Ilupeju'],
    'Ojodu':    ['Ojodu', 'Berger', 'Isheri', 'Magodo', 'Shangisha', 'Omole', 'Ketu', 'Ojota'],
    'Gbagada':  ['Gbagada', 'Soluyi', 'Ifako', 'Pedro', 'Ogudu'],
    'Yaba':     ['Yaba', 'Alagomeji', 'Akoka', 'Sabo', 'Abule Ijesha', 'Bariga', 'Shomolu'],
    'Surulere': ['Surulere', 'Ojuelegba', 'Aguda', 'Itire', 'Bode Thomas', 'Eric Moore'],
    'Isolo':    ['Isolo', 'Ajao Estate', 'Mafoluku', 'Oshodi', 'Ejigbo'],
    'Ikorodu':  ['Ikorodu', 'Ajuwon', 'Akute', 'Agric', 'Imota', 'Kosofe', 'Mile 12'],
    'Alimosho': ['Alimosho', 'Ipaja', 'Egbeda', 'Akesan', 'Abule Egba', 'Meiran', 'Agege'],
    'Okota':    ['Okota', 'Ago Palace'],
    'Amuwo':    ['Amuwo', 'Festac', 'Mile 2', 'Apapa', 'Kirikiri'],
    'Badagry':  ['Badagry', 'Okokomaiko', 'Iyana Iba'],
}


def get_consolidated_location(loc_string, loc_counts_dict):
    if loc_counts_dict.get(loc_string, 0) >= RARE_THRESHOLD:
        return loc_string
    loc_lower = loc_string.lower()
    for district, keywords in DISTRICT_MAP.items():
        if any(kw.lower() in loc_lower for kw in keywords):
            return district
    return 'Lagos Other'
# ============================================================
# [END ADDED]
# ============================================================


def engineer_features(input_csv):
    print("--- INITIATING DOMAIN-EXPERT FEATURE ENGINEERING (OHE + CONSOLIDATION) ---")

    df = pd.read_csv(input_csv)

    df['Property_Profile'] = (
        df['Raw_Description'].fillna('') + " " +
        df['Detailed_Attributes'].fillna('') + " " +
        df['Amenities'].fillna('')
    )

    print("Extracting verified Lagos real estate amenities via Regex...")
    df, extracted_feature_columns = extract_comprehensive_amenities(df, 'Property_Profile')

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

    # ============================================================
    # [ADDED] LOCATION CONSOLIDATION STEP
    # TO REVERT: Delete this block and change 'Consolidated_Location'
    # to 'Cleaned_Location' in features_to_keep and ColumnTransformer.
    # ============================================================
    print(f"Consolidating rare sub-locations (threshold = {RARE_THRESHOLD} rows)...")
    loc_counts_dict = df['Cleaned_Location'].value_counts().to_dict()
    df['Consolidated_Location'] = df['Cleaned_Location'].apply(
        lambda x: get_consolidated_location(x, loc_counts_dict)
    )
    unique_before = df['Cleaned_Location'].nunique()
    unique_after = df['Consolidated_Location'].nunique()
    print(f"  Location labels: {unique_before} -> {unique_after} unique values")
    # ============================================================
    # [END ADDED]
    # ============================================================

    # [MODIFIED] 'Consolidated_Location' replaces 'Cleaned_Location'.
    # TO REVERT: change back to 'Cleaned_Location'.
    features_to_keep = ['Consolidated_Location'] + extracted_feature_columns + structural_features
    X = df[features_to_keep]

    print("Applying np.log1p transformation to Target Variable (TCT)...")
    y = np.log1p(df['TCT'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Dataset Split: {len(X_train)} Training rows | {len(X_test)} Testing rows")

    # ============================================================
    # [REMOVED] ISOLATION FOREST (was in 4th attempt)
    # The commercial filter in outlier_removal_and_misc.py now handles
    # real anomalies upstream. Isolation Forest here would drop valid
    # residential properties unnecessarily.
    #
    # TO RESTORE: Uncomment and add IsolationForest import at top.
    # ------------------------------------------------------------
    # X_train_numeric = X_train.drop(columns=['Consolidated_Location'])
    # iso_forest = IsolationForest(contamination=0.05, random_state=42)
    # outlier_labels = iso_forest.fit_predict(X_train_numeric)
    # mask = outlier_labels != -1
    # X_train = X_train[mask]
    # y_train = y_train[mask]
    # ============================================================

    # [MODIFIED] Back to OneHotEncoder. The 4th attempt used TargetEncoder.
    # OHE is restored because consolidation above solves the sparsity problem,
    # and TargetEncoder scored worse (MAPE 82.78% vs OHE baseline 76.42%).
    # TO REVERT TO TARGETENCODER: swap the ColumnTransformer content and
    # restore the TargetEncoder import and .npy saves below.
    preprocessor = ColumnTransformer(
        transformers=[
            # [MODIFIED] Column changed from 'Cleaned_Location' to 'Consolidated_Location'
            ('location', OneHotEncoder(handle_unknown='ignore'), ['Consolidated_Location'])
        ],
        remainder='passthrough'
    )

    print("Fitting OneHotEncoder on consolidated location labels...")
    X_train_matrix = preprocessor.fit_transform(X_train)
    X_test_matrix = preprocessor.transform(X_test)

    print(f"Extraction complete. Feature dimensions: {X_train_matrix.shape[1]}")

    # [MODIFIED] Saving as .npz (sparse) — correct for OHE output.
    # The 4th attempt saved .npy (dense) because TargetEncoder output is dense.
    # Training scripts must use scipy.sparse.load_npz(), not np.load().
    print("\nSaving sparse matrices...")
    scipy.sparse.save_npz('X_train_processed.npz', X_train_matrix)
    scipy.sparse.save_npz('X_test_processed.npz', X_test_matrix)

    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

    joblib.dump(preprocessor, 'feature_preprocessor.pkl')

    print("Success. Files: X_train_processed.npz, X_test_processed.npz,")
    print("                y_train.csv, y_test.csv, feature_preprocessor.pkl")


if __name__ == "__main__":
    engineer_features('lagos_rent_tct_cleaned.csv')
























# Fourth attempt at the feature engineering pipeline. This introduces two new features: Isolation Forest for outlier detection and Target Encoding for location features. This version is designed to be more robust and better suited for model training, especially when dealing with the unique characteristics of Lagos property listings.
# import pandas as pd
# import numpy as np
# import re
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import IsolationForest
# from category_encoders import TargetEncoder
# from sklearn.compose import ColumnTransformer
# import joblib

# def extract_comprehensive_amenities(df, description_column):
#     # I am working on a copy so I don't accidentally ruin my raw dataset
#     data = df.copy()
    
#     data['clean_desc'] = data[description_column].fillna('').str.lower()
    
#     comprehensive_features = {
#         'has_prepaid_meter': r'(pre.?paid|personal meter)',
#         'has_screeded_walls': r'(screed)',
#         'has_fitted_kitchen': r'(fitted kitchen|kitchen cab|cabs)',
#         'has_clean_water': r'(clean water|bore hole|borehole|treated|water board)',
#         'has_steady_power': r'(24.?7|24.?hr|constant light|power supply|inverter|gen|solar)',
#         'has_ac': r'(a/?c|air.?con)',
#         'has_water_heater': r'(water heater|hot water)',
#         'has_internet': r'(wifi|wi-fi|internet|fiber|fibre)',
#         'is_ensuite': r'(en.?suite|all rooms en)',
#         'has_bq': r'(bq|boys quarter|maid)', 
#         'has_pop_ceiling': r'(pop|p\.o\.p)',
#         'has_wardrobe': r'(wardrobe|closet)',
#         'has_balcony': r'(balcony|terrace)',
#         'is_secured': r'(secur|guard|patrol|safe)',
#         'in_gated_estate': r'(gated|estate|call to enter)',
#         'has_cctv': r'(cctv|camera|surveillance)',
#         'has_pool': r'(pool|swimming)',
#         'has_gym': r'(gym|fitness)',
#         'has_smart_home': r'(smart|automated)',
#         'has_elevator': r'(elevator|lift)',
#         'has_parking': r'(park|garage)',
#         'is_tarred': r'(tarred|paved|interlock)',
#         'has_easy_access': r'(easy access|bus stop|accessible)',
#         'prefers_working_class': r'(working class|corporate)'
#     }
    
#     # Looping through my master dictionary to create binary columns
#     for feature_name, pattern in comprehensive_features.items():
#         data[feature_name] = data['clean_desc'].apply(
#             lambda x: 1 if re.search(pattern, x) else 0
#         )
        
#     data = data.drop(columns=['clean_desc'])
#     return data, list(comprehensive_features.keys())

# def engineer_features(input_csv):
#     print("--- INITIATING OPTIMIZED DOMAIN-EXPERT PIPELINE ---")
    
#     df = pd.read_csv(input_csv)
    
#     df['Property_Profile'] = df['Raw_Description'].fillna('') + " " + \
#                              df['Detailed_Attributes'].fillna('') + " " + \
#                              df['Amenities'].fillna('')
                             
#     print("Extracting verified Lagos real estate amenities via Regex...")
#     df, extracted_feature_columns = extract_comprehensive_amenities(df, 'Property_Profile')
    
#     # ==========================================
#     # EXTRACTING STRUCTURAL SIZING DATA
#     # ==========================================
#     print("Extracting property sizing (Bedrooms, Bathrooms, Serviced/Furnished)...")
    
#     df['Top_Icons'] = df['Top_Icons'].fillna('')
#     df['Raw_Description'] = df['Raw_Description'].fillna('').str.lower()
    
#     df['Bedrooms'] = df['Top_Icons'].str.extract(r'(\d+)\s*bedroom', flags=re.IGNORECASE)[0].fillna(0).astype(int)
#     desc_beds = df['Raw_Description'].str.extract(r'(\d+)\s*-?\s*(?:bedroom|bed|brm|bdrm|bedeoom|beds)\b')[0].fillna(0).astype(int)
#     df['Bedrooms'] = np.where(df['Bedrooms'] == 0, desc_beds, df['Bedrooms'])
    
#     implied_1_bed = df['Raw_Description'].str.contains(r'(?:mini\s*flat|self\s*con|room\s*and\s*parlour|room\s*&\s*parlour|selfcon|bq|room\s*self|roomself|single\s*room)', na=False)
#     df['Bedrooms'] = np.where((df['Bedrooms'] == 0) & implied_1_bed, 1, df['Bedrooms'])
    
#     df['Bathrooms'] = df['Top_Icons'].str.extract(r'(\d+)\s*bathroom', flags=re.IGNORECASE)[0].fillna(0).astype(int)
#     desc_baths = df['Raw_Description'].str.extract(r'(\d+)\s*(?:bath|bathroom|bathrooms|baths|toilets)\b')[0].fillna(0).astype(int)
#     df['Bathrooms'] = np.where(df['Bathrooms'] == 0, desc_baths, df['Bathrooms'])
    
#     df['Is_Serviced'] = df['Is_Serviced'].fillna(0).astype(int)
#     df['Is_Furnished'] = df['Is_Furnished'].fillna(0).astype(int)
    
#     structural_features = ['Bedrooms', 'Bathrooms', 'Is_Serviced', 'Is_Furnished']

#     # ==========================================

#     features_to_keep = ['Cleaned_Location'] + extracted_feature_columns + structural_features
#     X = df[features_to_keep]
    
#     print("Applying np.log1p transformation to Target Variable (TCT)...")
#     y = np.log1p(df['TCT'])
    
#     # Splitting the data first so I don't leak test data into my Target Encoder or Isolation Forest
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     print(f"Initial Dataset Split: {len(X_train)} Training rows | {len(X_test)} Testing rows")

#     # ==========================================
#     # PHASE 1: ISOLATION FOREST (THE BOUNCER)
#     # ==========================================
#     print("Running Isolation Forest to detect and drop mega-outliers in training data...")
    
#     # I only want to run this on numerical data, so I'll temporarily drop the location string
#     X_train_numeric = X_train.drop(columns=['Cleaned_Location'])
    
#     # Contamination is set to 0.05, meaning I am asking the algorithm to drop the weirdest 5% of properties
#     iso_forest = IsolationForest(contamination=0.05, random_state=42)
#     outlier_labels = iso_forest.fit_predict(X_train_numeric)
    
#     # Keeping only the properties labeled as '1' (normal). '-1' means anomaly.
#     mask = outlier_labels != -1
#     X_train = X_train[mask]
#     y_train = y_train[mask]
#     print(f"Dropped anomalies. New Training rows: {len(X_train)}")

#     # ==========================================
#     # PHASE 2: TARGET ENCODING (THE TRANSLATOR)
#     # ==========================================
#     # Replacing OneHotEncoder with TargetEncoder for my locations so I get continuous data, not sparse matrices
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('location', TargetEncoder(), ['Cleaned_Location'])
#         ],
#         remainder='passthrough'
#     )

#     print("Fitting Target Encoder... (Converting text locations into economic weights)")
    
#     # Notice I am fitting on X_train and y_train so it learns the relationship
#     X_train_matrix = preprocessor.fit_transform(X_train, y_train)
#     X_test_matrix = preprocessor.transform(X_test)
    
#     print(f"Extraction Complete. Dimensions per property: {X_train_matrix.shape[1]} features.")

#     print("\nSaving dense matrices for Optuna tuning...")
    
#     # I am saving these as standard numpy arrays now because Target Encoding makes the matrix dense, not sparse.
#     np.save('X_train_processed.npy', X_train_matrix)
#     np.save('X_test_processed.npy', X_test_matrix)
    
#     y_train.to_csv('y_train.csv', index=False)
#     y_test.to_csv('y_test.csv', index=False)
    
#     joblib.dump(preprocessor, 'feature_preprocessor.pkl')
    
#     print("Success. Outliers removed, Target Encoding applied, and files exported.")

# if __name__ == "__main__":
#     engineer_features('lagos_rent_tct_cleaned.csv')

























# (Best) Third attempt at a fixed feature engineering pipeline. This version applies domain-expert knowledge to extract specific, high-impact features from the text data, while also addressing the unique challenges of Lagos property listings. The code is designed to be robust and adaptable, allowing for dynamic adjustments to the feature extraction process based on the dataset's characteristics.
# import pandas as pd
# import numpy as np
# import scipy.sparse
# import re
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# import joblib

# def extract_comprehensive_amenities(df, description_column):
#     # I am working on a copy so I don't accidentally ruin my raw dataset
#     data = df.copy()
    
#     data['clean_desc'] = data[description_column].fillna('').str.lower()
    
#     comprehensive_features = {
#         'has_prepaid_meter': r'(pre.?paid|personal meter)',
#         'has_screeded_walls': r'(screed)',
#         'has_fitted_kitchen': r'(fitted kitchen|kitchen cab|cabs)',
#         'has_clean_water': r'(clean water|bore hole|borehole|treated|water board)',
#         'has_steady_power': r'(24.?7|24.?hr|constant light|power supply|inverter|gen|solar)',
#         'has_ac': r'(a/?c|air.?con)',
#         'has_water_heater': r'(water heater|hot water)',
#         'has_internet': r'(wifi|wi-fi|internet|fiber|fibre)',
#         'is_ensuite': r'(en.?suite|all rooms en)',
#         'has_bq': r'(bq|boys quarter|maid)', 
#         'has_pop_ceiling': r'(pop|p\.o\.p)',
#         'has_wardrobe': r'(wardrobe|closet)',
#         'has_balcony': r'(balcony|terrace)',
#         'is_secured': r'(secur|guard|patrol|safe)',
#         'in_gated_estate': r'(gated|estate|call to enter)',
#         'has_cctv': r'(cctv|camera|surveillance)',
#         'has_pool': r'(pool|swimming)',
#         'has_gym': r'(gym|fitness)',
#         'has_smart_home': r'(smart|automated)',
#         'has_elevator': r'(elevator|lift)',
#         'has_parking': r'(park|garage)',
#         'is_tarred': r'(tarred|paved|interlock)',
#         'has_easy_access': r'(easy access|bus stop|accessible)',
#         'prefers_working_class': r'(working class|corporate)'
#     }
    
#     for feature_name, pattern in comprehensive_features.items():
#         data[feature_name] = data['clean_desc'].apply(
#             lambda x: 1 if re.search(pattern, x) else 0
#         )
        
#     data = data.drop(columns=['clean_desc'])
#     return data, list(comprehensive_features.keys())


# def engineer_features(input_csv):
#     print("--- INITIATING DOMAIN-EXPERT FEATURE ENGINEERING ---")
    
#     df = pd.read_csv(input_csv)
    
#     df['Property_Profile'] = df['Raw_Description'].fillna('') + " " + \
#                              df['Detailed_Attributes'].fillna('') + " " + \
#                              df['Amenities'].fillna('')
                             
#     print("Extracting verified Lagos real estate amenities via Regex...")
#     df, extracted_feature_columns = extract_comprehensive_amenities(df, 'Property_Profile')
    
#     # ==========================================
#     # NEW: EXTRACTING STRUCTURAL SIZING DATA
#     # ==========================================
#     print("Extracting property sizing (Bedrooms, Bathrooms, Serviced/Furnished)...")
    
#     # 1. Clean the target columns for regex scanning
#     df['Top_Icons'] = df['Top_Icons'].fillna('')
#     df['Raw_Description'] = df['Raw_Description'].fillna('').str.lower()
    
#     # 2. Extract Bedrooms (Tries Top_Icons first, then falls back to descriptions)
#     df['Bedrooms'] = df['Top_Icons'].str.extract(r'(\d+)\s*bedroom', flags=re.IGNORECASE)[0].fillna(0).astype(int)
#     desc_beds = df['Raw_Description'].str.extract(r'(\d+)\s*-?\s*(?:bedroom|bed|brm|bdrm|bedeoom|beds)\b')[0].fillna(0).astype(int)
#     df['Bedrooms'] = np.where(df['Bedrooms'] == 0, desc_beds, df['Bedrooms'])
    
#     # Catching implied 1-bedrooms like "mini flat", "self con", "roomself"
#     implied_1_bed = df['Raw_Description'].str.contains(r'(?:mini\s*flat|self\s*con|room\s*and\s*parlour|room\s*&\s*parlour|selfcon|bq|room\s*self|roomself|single\s*room)', na=False)
#     df['Bedrooms'] = np.where((df['Bedrooms'] == 0) & implied_1_bed, 1, df['Bedrooms'])
    
#     # 3. Extract Bathrooms
#     df['Bathrooms'] = df['Top_Icons'].str.extract(r'(\d+)\s*bathroom', flags=re.IGNORECASE)[0].fillna(0).astype(int)
#     desc_baths = df['Raw_Description'].str.extract(r'(\d+)\s*(?:bath|bathroom|bathrooms|baths|toilets)\b')[0].fillna(0).astype(int)
#     df['Bathrooms'] = np.where(df['Bathrooms'] == 0, desc_baths, df['Bathrooms'])
    
#     # 4. Fill in missing boolean flags
#     df['Is_Serviced'] = df['Is_Serviced'].fillna(0).astype(int)
#     df['Is_Furnished'] = df['Is_Furnished'].fillna(0).astype(int)
    
#     structural_features = ['Bedrooms', 'Bathrooms', 'Is_Serviced', 'Is_Furnished']

#     # ==========================================

#     # Combine Location + Amenities + Structural Data
#     features_to_keep = ['Cleaned_Location'] + extracted_feature_columns + structural_features
#     X = df[features_to_keep]
    
#     print("Applying np.log1p transformation to Target Variable (TCT)...")
#     y = np.log1p(df['TCT'])
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     print(f"Dataset Split: {len(X_train)} Training rows | {len(X_test)} Testing rows")

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('location', OneHotEncoder(handle_unknown='ignore'), ['Cleaned_Location'])
#         ],
#         remainder='passthrough'
#     )

#     print("Fitting Encoder... (Preparing final matrices)")
#     X_train_matrix = preprocessor.fit_transform(X_train)
#     X_test_matrix = preprocessor.transform(X_test)
    
#     print(f"Extraction Complete. Dimensions per property: {X_train_matrix.shape[1]} features.")

#     print("\nSaving matrices for model training...")
#     scipy.sparse.save_npz('X_train_processed.npz', X_train_matrix)
#     scipy.sparse.save_npz('X_test_processed.npz', X_test_matrix)
    
#     y_train.to_csv('y_train.csv', index=False)
#     y_test.to_csv('y_test.csv', index=False)
    
#     joblib.dump(preprocessor, 'feature_preprocessor.pkl')
    
#     print("Success. Clean matrices generated with architectural size data.")

# if __name__ == "__main__":
#     engineer_features('lagos_rent_tct_cleaned.csv')















# Second attempt at a fixed feature engineering pipeline. This version applies domain-expert knowledge to extract specific, high-impact features from the text data, while also addressing the unique challenges of Lagos property listings. The code is designed to be robust and adaptable, allowing for dynamic adjustments to the feature extraction process based on the dataset's characteristics.
# import pandas as pd
# import numpy as np
# import scipy.sparse
# import re
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# import joblib

# def extract_comprehensive_amenities(df, description_column):
#     # I am working on a copy so I don't accidentally ruin my raw dataset
#     data = df.copy()
    
#     # Cleaning the text column first. Lowercase makes matching easier, 
#     # and filling NaNs prevents the regex engine from throwing errors.
#     data['clean_desc'] = data[description_column].fillna('').str.lower()
    
#     # This is my master dictionary. It combines general luxury features with 
#     # high-impact Lagos-specific features. I've optimized the patterns to be 
#     # highly flexible, catching root words rather than rigid exact phrases.
#     comprehensive_features = {
#         'has_prepaid_meter': r'(pre.?paid|personal meter)',
#         'has_screeded_walls': r'(screed)',
#         'has_fitted_kitchen': r'(fitted kitchen|kitchen cab|cabs)',
#         'has_clean_water': r'(clean water|bore hole|borehole|treated|water board)',
#         'has_steady_power': r'(24.?7|24.?hr|constant light|power supply|inverter|gen|solar)',
#         'has_ac': r'(a/?c|air.?con)',
#         'has_water_heater': r'(water heater|hot water)',
#         'has_internet': r'(wifi|wi-fi|internet|fiber|fibre)',
#         'is_ensuite': r'(en.?suite|all rooms en)',
#         'has_bq': r'(bq|boys quarter|maid)', 
#         'has_pop_ceiling': r'(pop|p\.o\.p)',
#         'has_wardrobe': r'(wardrobe|closet)',
#         'has_balcony': r'(balcony|terrace)',
#         'is_secured': r'(secur|guard|patrol|safe)',
#         'in_gated_estate': r'(gated|estate|call to enter)',
#         'has_cctv': r'(cctv|camera|surveillance)',
#         'has_pool': r'(pool|swimming)',
#         'has_gym': r'(gym|fitness)',
#         'has_smart_home': r'(smart|automated)',
#         'has_elevator': r'(elevator|lift)',
#         'has_parking': r'(park|garage)',
#         'is_tarred': r'(tarred|paved|interlock)',
#         'is_serviced': r'(servic)', 
#         'has_easy_access': r'(easy access|bus stop|accessible)',
#         'prefers_working_class': r'(working class|corporate)'
#     }
    
#     # I am looping through my master dictionary to create binary columns.
#     # 1 if the flexible pattern finds a match, 0 if it doesn't.
#     for feature_name, pattern in comprehensive_features.items():
#         data[feature_name] = data['clean_desc'].apply(
#             lambda x: 1 if re.search(pattern, x) else 0
#         )
        
#     # Dropping the temporary text column to keep my dataframe clean for modeling
#     data = data.drop(columns=['clean_desc'])
    
#     # I am returning the modified dataframe and a list of the newly created feature names
#     return data, list(comprehensive_features.keys())


# def engineer_features(input_csv):
#     print("--- INITIATING DOMAIN-EXPERT FEATURE ENGINEERING ---")
    
#     # 1. Loading my clean dataset
#     df = pd.read_csv(input_csv)
    
#     # 2. Consolidating my text columns into a single profile for the regex engine to scan
#     df['Property_Profile'] = df['Raw_Description'].fillna('') + " " + \
#                              df['Detailed_Attributes'].fillna('') + " " + \
#                              df['Amenities'].fillna('')
                             
#     # 3. Running my hardcoded extraction function to pull out the valid structural drivers
#     print("Extracting verified Lagos real estate features via Regex...")
#     df, extracted_feature_columns = extract_comprehensive_amenities(df, 'Property_Profile')
    
#     # 4. Defining my feature matrix (X). It includes the location and all my new binary columns.
#     features_to_keep = ['Cleaned_Location'] + extracted_feature_columns
#     X = df[features_to_keep]
    
#     # 5. Applying the Logarithmic Transformation to my target variable to fix the mean bias (MAPE issue)
#     print("Applying np.log1p transformation to Target Variable (TCT)...")
#     y = np.log1p(df['TCT'])
    
#     # 6. Splitting my dataset. I'm using 80/20 to give my models enough training data.
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     print(f"Dataset Split: {len(X_train)} Training rows | {len(X_test)} Testing rows")

#     # 7. Building my preprocessor. I only need to One-Hot Encode the location now.
#     # By using remainder='passthrough', my preprocessor will let the binary 1/0 columns flow straight to the model.
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('location', OneHotEncoder(handle_unknown='ignore'), ['Cleaned_Location'])
#         ],
#         remainder='passthrough'
#     )

#     # 8. Fitting my encoder and transforming the data
#     print("Fitting Encoder... (Preparing final matrices)")
#     X_train_matrix = preprocessor.fit_transform(X_train)
#     X_test_matrix = preprocessor.transform(X_test)
    
#     print(f"Extraction Complete. Dimensions per property: {X_train_matrix.shape[1]} features.")

#     # 9. Exporting my matrices, target variables, and the preprocessor pipeline
#     print("\nSaving matrices for model training...")
#     scipy.sparse.save_npz('X_train_processed.npz', X_train_matrix)
#     scipy.sparse.save_npz('X_test_processed.npz', X_test_matrix)
    
#     y_train.to_csv('y_train.csv', index=False)
#     y_test.to_csv('y_test.csv', index=False)
    
#     joblib.dump(preprocessor, 'feature_preprocessor.pkl')
    
#     print("Success. Clean, hardcoded feature matrices and log-transformed targets generated.")

# if __name__ == "__main__":
#     engineer_features('lagos_rent_tct_cleaned.csv')













# First attempt at a fixed feature engineering pipeline. This version applies mathematical filtering techniques to ensure that only meaningful features are extracted from the text data, while also addressing the unique challenges of Lagos property listings. The code is designed to be robust and adaptable, allowing for dynamic adjustments to the feature extraction process based on the dataset's characteristics.
# import pandas as pd
# import scipy.sparse
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# import joblib

# def engineer_features(input_csv):
#     print("--- INITIATING STATISTICAL FEATURE ENGINEERING ---")
    
#     # Load my clean dataset
#     df = pd.read_csv(input_csv)
    
#     # Consolidate my text columns into a single profile for the algorithm to scan
#     df['Property_Profile'] = df['Raw_Description'].fillna('') + " " + \
#                              df['Detailed_Attributes'].fillna('') + " " + \
#                              df['Amenities'].fillna('')
                             
#     X = df[['Cleaned_Location', 'Property_Profile']]
#     y = df['TCT']
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     print(f"Dataset Split: {len(X_train)} Training rows | {len(X_test)} Testing rows")

#     # I'm keeping a small list of Lagos-specific agent spam to supplement the standard English stop words
#     lagos_spam_words = [
#         'call', 'whatsapp', 'agent', 'inspection', 'fee', 'naira', 'million', 
#         'pay', 'rent', 'contact', 'properties', 'property', 'letting', 
#         'lease', 'year', 'annum', 'monthly', 'details', 'dm', 'available'
#     ]
#     custom_stop_words = list(ENGLISH_STOP_WORDS) + lagos_spam_words

#     # Building the preprocessor using statistical pruning instead of hardcoding
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('location', OneHotEncoder(handle_unknown='ignore'), ['Cleaned_Location']),
#             ('text', TfidfVectorizer(
#                 max_features=400, 
#                 stop_words=custom_stop_words, 
#                 ngram_range=(1, 2),
#                 # MATHEMATICAL FILTERING APPLIED HERE:
#                 # 1. token_pattern: Only accept words with alphabet characters (a-z) that are 2+ letters long.
#                 # This naturally drops '10', '000 000', and single letter typos.
#                 token_pattern=r'(?u)\b[a-zA-Z]{2,}\b',
#                 # 2. min_df: Ignore terms that appear in less than 1% of the properties (Kills typos)
#                 min_df=0.01,
#                 # 3. max_df: Ignore terms that appear in more than 80% of properties (Kills ubiquitous spam)
#                 max_df=0.80 
#             ), 'Property_Profile')
#         ])

#     print("Fitting Vectorizer and Encoders... (Applying statistical thresholds)")
#     X_train_matrix = preprocessor.fit_transform(X_train)
#     X_test_matrix = preprocessor.transform(X_test)
    
#     print(f"Extraction Complete. Dimensions per property: {X_train_matrix.shape[1]} features.")

#     print("\nSaving math matrices for model training...")
#     scipy.sparse.save_npz('X_train_processed.npz', X_train_matrix)
#     scipy.sparse.save_npz('X_test_processed.npz', X_test_matrix)
    
#     y_train.to_csv('y_train.csv', index=False)
#     y_test.to_csv('y_test.csv', index=False)
    
#     joblib.dump(preprocessor, 'feature_preprocessor.pkl')
    
#     print("Success. Dynamically cleaned matrices generated.")

# if __name__ == "__main__":
#     engineer_features('lagos_rent_tct_cleaned.csv')



# Flawed version of the feature engineering pipeline. This version contains critical errors that will lead to incorrect feature extraction and model training. Do NOT use this code for any production or training purposes.
# import pandas as pd
# import scipy.sparse
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# import joblib

# def engineer_features(input_csv):
#     print("--- INITIATING FEATURE ENGINEERING PIPELINE ---")
    
#     # 1. Load the dataset
#     df = pd.read_csv(input_csv)
    
#     # Drop properties where TCT is missing or 0 to ensure clean training
#     df = df[df['TCT'] > 0].reset_index(drop=True)
    
#     # 2. Consolidate Text Features
#     df['Property_Profile'] = df['Raw_Description'].fillna('') + " " + \
#                              df['Detailed_Attributes'].fillna('') + " " + \
#                              df['Amenities'].fillna('')
                             
#     # 3. Define X (Features) and y (Target)
#     # Corrected to use 'Cleaned_Location'
#     X = df[['Cleaned_Location', 'Property_Profile']]
#     y = df['TCT']
    
#     # 4. Strict Train-Test Split (80/20)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     print(f"Dataset Split: {len(X_train)} Training rows | {len(X_test)} Testing rows")

#     # 5. Build the Preprocessor
#     # Corrected to target 'Cleaned_Location'
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('location', OneHotEncoder(handle_unknown='ignore'), ['Cleaned_Location']),
#             ('text', TfidfVectorizer(max_features=400, stop_words='english', ngram_range=(1, 2)), 'Property_Profile')
#         ])

#     # 6. Fit and Transform
#     print("Fitting Vectorizer and Encoders... (Extracting hidden features)")
#     X_train_matrix = preprocessor.fit_transform(X_train)
#     X_test_matrix = preprocessor.transform(X_test)
    
#     print(f"Extraction Complete. Dimensions per property: {X_train_matrix.shape[1]} features.")

#     # 7. Export the Matrices and Target Variables
#     print("\nSaving mathematical matrices for model training...")
#     scipy.sparse.save_npz('X_train_processed.npz', X_train_matrix)
#     scipy.sparse.save_npz('X_test_processed.npz', X_test_matrix)
    
#     y_train.to_csv('y_train.csv', index=False)
#     y_test.to_csv('y_test.csv', index=False)
    
#     # Save the fitted preprocessor so we can use it later in the FastAPI backend
#     joblib.dump(preprocessor, 'feature_preprocessor.pkl')
    
#     print("Success. Saved 'X_train_processed.npz', 'X_test_processed.npz', 'y_train.csv', 'y_test.csv', and 'feature_preprocessor.pkl'.")

# if __name__ == "__main__":
#     engineer_features('lagos_rent_tct_cleaned.csv')