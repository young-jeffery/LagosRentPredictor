import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
import joblib

# I am keeping my rare location threshold at 5 to ensure my target encoder has enough rows to calculate a reliable mean
RARE_THRESHOLD = 5

DISTRICT_MAP = {
    'Lekki':    ['Lekki', 'Ikate', 'Agungi', 'Chevron', 'Osapa', 'Ikota', 'Marwa', 'Igbo Efon'],
    'Ajah':     ['Ajah', 'Sangotedo', 'Badore', 'Awoyaya', 'Bogije', 'Abraham Adesanya', 'Ibeju', 'Eleko', 'Epe', 'Ado'],
    'VI':       ['Victoria Island', 'Adeola', 'Akin Adesola', 'Ahmadu Bello', 'Ozumba'],
    'Ikoyi':    ['Ikoyi', 'Osborne', 'Parkview', 'Old Ikoyi', 'Banana Island'],
    'Ikeja':    ['Ikeja', 'Allen', 'Alausa', 'Oregun', 'Adeniyi Jones', 'Toyin', 'Acme', 'Ogba', 'Agidingbi'],
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
    # I am checking if the location appears enough times to be statistically significant
    if loc_counts_dict.get(loc_string, 0) >= RARE_THRESHOLD:
        return loc_string
    loc_lower = loc_string.lower()
    for district, keywords in DISTRICT_MAP.items():
        if any(kw.lower() in loc_lower for kw in keywords):
            return district
    return 'Lagos Other'

def extract_comprehensive_amenities(df, description_column):
    # I am making a copy of my dataframe so I don't alter the raw data directly
    data = df.copy()
    data['clean_desc'] = data[description_column].fillna('').str.lower()
    
    comprehensive_features = {
        'has_prepaid_meter': r'(pre.?paid|personal meter)',
        'has_screeded_walls': r'(screed)',
        'has_fitted_kitchen': r'(fitted kitchen|kitchen cab|cabs)',
        'has_clean_water': r'(clean water|bore hole|borehole|treated|water board)',
        'has_steady_power': r'(24.?7|24.?hr|constant light|power supply|inverter|gen|solar)',
        'has_ac': r'(a/?c|air.?con)',
        'has_water_heater': r'(water heater|hot water)',
        'has_internet': r'(wifi|wi-fi|internet|fiber|fibre)',
        'is_ensuite': r'(en.?suite|all rooms en)',
        'has_bq': r'(bq|boys quarter|maid)', 
        'has_pop_ceiling': r'(pop|p\.o\.p)',
        'has_wardrobe': r'(wardrobe|closet)',
        'has_balcony': r'(balcony|terrace)',
        'is_secured': r'(secur|guard|patrol|safe)',
        'in_gated_estate': r'(gated|estate|call to enter)',
        'has_cctv': r'(cctv|camera|surveillance)',
        'has_pool': r'(pool|swimming)',
        'has_gym': r'(gym|fitness)',
        'has_smart_home': r'(smart|automated)',
        'has_elevator': r'(elevator|lift)',
        'has_parking': r'(park|garage)',
        'is_tarred': r'(tarred|paved|interlock)',
        'has_easy_access': r'(easy access|bus stop|accessible)',
        'prefers_working_class': r'(working class|corporate)'
    }
    
    # I am generating my binary flags based on my regex dictionary
    for feature_name, pattern in comprehensive_features.items():
        data[feature_name] = data['clean_desc'].apply(
            lambda x: 1 if re.search(pattern, x) else 0
        )
        
    data = data.drop(columns=['clean_desc'])
    return data, list(comprehensive_features.keys())

def engineer_features_dense(input_csv):
    print("--- INITIATING DENSE TARGET-ENCODED FEATURE ENGINEERING ---")
    
    df = pd.read_csv(input_csv)
    
    df['Property_Profile'] = df['Raw_Description'].fillna('') + " " + \
                             df['Detailed_Attributes'].fillna('') + " " + \
                             df['Amenities'].fillna('')
                             
    print("Extracting verified Lagos real estate amenities...")
    df, extracted_feature_columns = extract_comprehensive_amenities(df, 'Property_Profile')
    
    print("Extracting property sizing...")
    df['Top_Icons'] = df['Top_Icons'].fillna('')
    df['Raw_Description'] = df['Raw_Description'].fillna('').str.lower()
    
    df['Bedrooms'] = df['Top_Icons'].str.extract(r'(\d+)\s*bedroom', flags=re.IGNORECASE)[0].fillna(0).astype(int)
    desc_beds = df['Raw_Description'].str.extract(r'(\d+)\s*-?\s*(?:bedroom|bed|brm|bdrm|bedeoom|beds)\b')[0].fillna(0).astype(int)
    df['Bedrooms'] = np.where(df['Bedrooms'] == 0, desc_beds, df['Bedrooms'])
    
    implied_1_bed = df['Raw_Description'].str.contains(r'(?:mini\s*flat|self\s*con|room\s*and\s*parlour|room\s*&\s*parlour|selfcon|bq|room\s*self|roomself|single\s*room)', na=False)
    df['Bedrooms'] = np.where((df['Bedrooms'] == 0) & implied_1_bed, 1, df['Bedrooms'])
    
    df['Bathrooms'] = df['Top_Icons'].str.extract(r'(\d+)\s*bathroom', flags=re.IGNORECASE)[0].fillna(0).astype(int)
    desc_baths = df['Raw_Description'].str.extract(r'(\d+)\s*(?:bath|bathroom|bathrooms|baths|toilets)\b')[0].fillna(0).astype(int)
    df['Bathrooms'] = np.where(df['Bathrooms'] == 0, desc_baths, df['Bathrooms'])
    
    df['Is_Serviced'] = df['Is_Serviced'].fillna(0).astype(int)
    df['Is_Furnished'] = df['Is_Furnished'].fillna(0).astype(int)
    
    structural_features = ['Bedrooms', 'Bathrooms', 'Is_Serviced', 'Is_Furnished']

    print(f"Consolidating rare sub-locations...")
    loc_counts_dict = df['Cleaned_Location'].value_counts().to_dict()
    df['Consolidated_Location'] = df['Cleaned_Location'].apply(
        lambda x: get_consolidated_location(x, loc_counts_dict)
    )

    features_to_keep = ['Consolidated_Location'] + extracted_feature_columns + structural_features
    X = df[features_to_keep]
    
    y = np.log1p(df['TCT'])
    
    # I must split my data before target encoding to prevent data leakage from the test set into my training weights
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # I am replacing OneHotEncoder with TargetEncoder because my Neural Network learns better from continuous dense data
    preprocessor = ColumnTransformer(
        transformers=[
            ('location', TargetEncoder(), ['Consolidated_Location'])
        ],
        remainder='passthrough'
    )

    print("Fitting Target Encoder on consolidated location labels...")
    X_train_matrix = preprocessor.fit_transform(X_train, y_train)
    X_test_matrix = preprocessor.transform(X_test)
    
    print(f"Extraction complete. Dense feature dimensions: {X_train_matrix.shape[1]}")

    print("\nSaving dense numpy arrays for ANN training...")
    # I am saving these as standard .npy files since they are fully dense now
    np.save('X_train_dense.npy', X_train_matrix)
    np.save('X_test_dense.npy', X_test_matrix)
    
    y_train.to_csv('y_train_dense.csv', index=False)
    y_test.to_csv('y_test_dense.csv', index=False)
    
    print("Success. Dense files exported.")

if __name__ == "__main__":
    engineer_features_dense('lagos_rent_tct_cleaned.csv')