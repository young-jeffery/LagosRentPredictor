import pandas as pd
import re

def extract_comprehensive_amenities(df, description_column):
    # I am working on a copy so I don't accidentally ruin my raw dataset
    data = df.copy()
    
    # Cleaning the text column first. Lowercase makes matching easier, 
    # and filling NaNs prevents the regex engine from throwing errors.
    data['clean_desc'] = data[description_column].fillna('').str.lower()
    
    # This is my master dictionary. It combines general luxury features with 
    # high-impact Lagos-specific features. I've optimized the patterns to be 
    # highly flexible, catching root words rather than rigid exact phrases.
    comprehensive_features = {
        # --- High-Value Lagos Essentials ---
        # Catching 'prepaid', 'pre-paid', or just 'personal meter'
        'has_prepaid_meter': r'(pre.?paid|personal meter)',
        
        # Catching 'screed', 'screeded', 'screeding'
        'has_screeded_walls': r'(screed)',
        
        # Catching 'kitchen cabinet', 'cabs', 'fitted kitchen'
        'has_fitted_kitchen': r'(fitted kitchen|kitchen cab|cabs)',
        
        # 'water board', 'borehole', 'clean water', 'treated'
        'has_clean_water': r'(clean water|bore hole|borehole|treated|water board)',
        
        # --- General Utilities & Power ---
        # Catching '24/7', '24hrs', 'generator', 'inverter', 'solar'
        'has_steady_power': r'(24.?7|24.?hr|constant light|power supply|inverter|gen|solar)',
        'has_ac': r'(a/?c|air.?con)',
        'has_water_heater': r'(water heater|hot water)',
        'has_internet': r'(wifi|wi-fi|internet|fiber|fibre)',
        
        # --- Structural & Space Layout ---
        # 'ensuite', 'en-suite', 'all rooms en'
        'is_ensuite': r'(en.?suite|all rooms en)',
        
        # Very specific to the Nigerian market
        'has_bq': r'(bq|boys quarter|maid)', 
        
        'has_pop_ceiling': r'(pop|p\.o\.p)',
        'has_wardrobe': r'(wardrobe|closet)',
        'has_balcony': r'(balcony|terrace)',
        
        # --- Security & Community ---
        'is_secured': r'(secur|guard|patrol|safe)',
        'in_gated_estate': r'(gated|estate|call to enter)',
        'has_cctv': r'(cctv|camera|surveillance)',
        
        # --- Luxury Add-ons ---
        'has_pool': r'(pool|swimming)',
        'has_gym': r'(gym|fitness)',
        'has_smart_home': r'(smart|automated)',
        'has_elevator': r'(elevator|lift)',
        
        # --- Exterior & Accessibility ---
        'has_parking': r'(park|garage)',
        'is_tarred': r'(tarred|paved|interlock)',
        'is_serviced': r'(servic)', # Catches 'serviced', 'service charge', 'services'
        'has_easy_access': r'(easy access|bus stop|accessible)',
        
        # --- Tenant Profile ---
        # If landlords restrict to corporate/working class, it usually indicates a premium tier
        'prefers_working_class': r'(working class|corporate)'
    }
    
    # I am looping through my master dictionary to create binary columns.
    # 1 if the flexible pattern finds a match, 0 if it doesn't.
    for feature_name, pattern in comprehensive_features.items():
        data[feature_name] = data['clean_desc'].apply(
            lambda x: 1 if re.search(pattern, x) else 0
        )
        
    # Dropping the temporary text column to keep my dataframe clean for modeling
    data = data.drop(columns=['clean_desc'])
    
    return data

# ==========================================
# How I will execute this in my workflow
# ==========================================
# df = pd.read_csv('lagos_rent_tct_cleaned.csv')
# df_fully_engineered = extract_comprehensive_amenities(df, 'Raw_Description')

# I can check how many properties actually have these features to ensure my regex worked
# print(df_fully_engineered[['has_screeded_walls', 'has_prepaid_meter', 'has_bq']].sum())