############## Block 1 ##############
import pandas as pd

# 1. Load both datasets
try:
    df_prop = pd.read_csv("propertypro_lagos_rent_data.csv")
    df_prop['Source'] = 'PropertyPro'
except FileNotFoundError:
    print("PropertyPro data not found. Check filename.")
    df_prop = pd.DataFrame()

try:
    df_jiji = pd.read_csv("jiji_property_rent_data.csv")
    df_jiji['Source'] = 'Jiji'
except FileNotFoundError:
    print("Jiji data not found. Check filename.")
    df_jiji = pd.DataFrame()

# 2. Combine the datasets
# Pandas will automatically align columns with the same names (like 'Price', 'Location', 'URL', 'Raw_Description')
df_combined = pd.concat([df_prop, df_jiji], ignore_index=True)

# 3. Duplicate Check
initial_rows = len(df_combined)
df_combined = df_combined.drop_duplicates(subset=['URL'], keep='first')
dropped_duplicates = initial_rows - len(df_combined)

# 4. Generate the Health Report
print("====== DATA HEALTH REPORT ======")
print(f"Total Unique Properties: {len(df_combined)}")
print(f"Duplicates Removed: {dropped_duplicates}")

print("\n--- Missing Data Breakdown ---")
# This shows how many blank cells exist in each column
missing_data = df_combined.isnull().sum()
print(missing_data[missing_data > 0]) 

print("\n--- Top 10 Locations (The Spread Check) ---")
print(df_combined['Location'].value_counts().head(10))

# 5. Export to Excel for Easy Viewing
excel_filename = "Combined_Raw_Data_View.xlsx"
df_combined.to_excel(excel_filename, index=False)
print(f"\nSUCCESS: Clean viewing file saved as '{excel_filename}'. You can open this in Excel or Google Sheets!")





############## Block 2 ##############
import pandas as pd
import re
from sklearn.cluster import KMeans
import numpy as np

# 1. Clean the Jiji Timestamps
# This regex looks for a comma, followed by a number, followed by hour/day/week/month, and removes it
time_pattern = r',\s*\d+\s*(second|minute|hour|day|week|month|year)s?\s*ago\s*$'
df_combined['Location'] = df_combined['Location'].str.replace(time_pattern, '', regex=True, flags=re.IGNORECASE)

# 2. Filter out Non-Residential Noise
# We drop rows if the description or icons mention these red-flag keywords
noise_keywords = 'block of flat|shortlet|commercial|office|shop|warehouse|hotel|per day|daily'
desc_mask = df_combined['Raw_Description'].fillna('').str.contains(noise_keywords, case=False)
icon_mask = df_combined['Top_Icons'].fillna('').str.contains(noise_keywords, case=False)

# Keep only the rows that DO NOT (~) have those noise keywords
df_cleaned = df_combined[~(desc_mask | icon_mask)].copy()
print(f"Dropped {len(df_combined) - len(df_cleaned)} noisy/non-residential records.")

# 3. Quick Numerical Price Conversion (Required for Segmentation Math)
# Strip out "₦", commas, and text to get pure integers
df_cleaned['Price_Num'] = df_cleaned['Price'].astype(str).str.replace(r'[^\d]', '', regex=True)
df_cleaned['Price_Num'] = pd.to_numeric(df_cleaned['Price_Num'], errors='coerce')
df_cleaned = df_cleaned.dropna(subset=['Price_Num']) # Drop rows where price was "Contact Agent"

# 4. Mathematical Market Segmentation (K-Means Clustering FIXED)
# Calculate the median price for every single location
location_medians = df_cleaned.groupby('Location')['Price_Num'].median().reset_index()

# APPLY LOG TRANSFORMATION: This compresses the massive billionaires-row outliers 
# so K-Means can actually see the difference between a 1M and a 10M property
location_medians['Log_Price'] = np.log1p(location_medians['Price_Num'])

# Use K-Means on the LOG prices
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
location_medians['Market_Tier'] = kmeans.fit_predict(location_medians[['Log_Price']])

# Figure out which cluster is the expensive one
cluster_centers = kmeans.cluster_centers_.flatten()
primary_cluster_id = np.argmax(cluster_centers)

# Label them clearly
location_medians['Market_Segment'] = np.where(
    location_medians['Market_Tier'] == primary_cluster_id, 
    'Primary (High-Income)', 
    'Secondary (Medium/Low-Income)'
)

# Merge the mathematical tiers back into our main dataset
df_cleaned = df_cleaned.merge(location_medians[['Location', 'Market_Segment']], on='Location', how='left')

# 5. Downsampling the Imbalance (The Bulletproof Way)
CAP = 250 

# Step A: Shuffle the entire dataset to ensure we get a random selection
df_shuffled = df_cleaned.sample(frac=1, random_state=42)

# Step B: Group by Location and take up to 'CAP' records per location
# This bypasses the buggy .apply() method entirely and keeps all columns safe.
df_balanced = df_shuffled.groupby('Location').head(CAP).reset_index(drop=True)

print("\n====== AFTER CLEANING & SEGMENTATION ======")
print(f"Total Properties Remaining: {len(df_balanced)}")

print("\n--- Market Segmentation Proof ---")
print(df_balanced['Market_Segment'].value_counts())

print("\n--- New Top 10 Locations (Capped at 250) ---")
print(df_balanced['Location'].value_counts().head(10))

# Export to view the results!
df_balanced.to_excel("Balanced_Segmented_Data.xlsx", index=False)





############### Block 3 ##############
import pandas as pd

# 1. PHASED GEOFENCING: Flagging Non-Lagos Properties
print("--- SUSPICIOUS LOCATION DETECTOR (AUTO-LAGOS FILTER) ---")

# List of common states/cities that might slip into the data
other_states = ['abuja', 'ogun', 'ibadan', 'oyo', 'rivers', 'portharcourt', 'kano', 'enugu', 'delta', 'kwara', 'fct', 'osun']

# Step A: Find locations containing other state keywords
contains_other_state = df_combined['Location'].str.lower().str.contains('|'.join(other_states), na=False)

# Step B: Find locations that explicitly mention Lagos
contains_lagos = df_combined['Location'].str.lower().str.contains('lagos', na=False)

# Step C: The True Suspicious List (Has another state AND DOES NOT have Lagos)
suspicious_mask = contains_other_state & ~contains_lagos
suspicious_locations = df_combined[suspicious_mask]['Location'].unique()

print(f"Found {len(suspicious_locations)} suspicious locations (Auto-approved any containing 'Lagos'):")
for loc in suspicious_locations:
    print(f" - {loc}")

print("\n(Action Required: If any of these are STILL actually in Lagos, put them in your 'valid_lagos_exceptions' list in the next block. Otherwise, leave it empty and they will be deleted.)")






############### Block 4 ##############
# --- EXECUTE DELETION (EXCEPTION BASED) & NORMALIZE LOCATIONS ---

# 1. Put the locations from the suspicious list that are ACTUALLY IN LAGOS here.
# The script will automatically delete everything else that was flagged.
valid_lagos_exceptions = [
    # Example: "Oyo Street Victoria Island",
    # Example: "Kwara Compound Lagos Island",
]

# Calculate exactly what to drop (Flagged items MINUS your exceptions)
locations_to_drop = [loc for loc in suspicious_locations if loc not in valid_lagos_exceptions]

print(f"Automatically dropping {len(locations_to_drop)} non-Lagos locations...")

# Drop them from the dataset
df_combined = df_combined[~df_combined['Location'].isin(locations_to_drop)]

# 2. ALPHABETICAL NORMALIZATION: Fixing "Ikate Lekki" vs "Lekki Ikate"
def standardize_location(loc_string):
    if pd.isna(loc_string): 
        return "Unknown"
    
    # Lowercase, remove punctuation, and remove the word "lagos"
    clean_str = re.sub(r'[^\w\s]', '', str(loc_string).lower())
    clean_str = clean_str.replace('lagos', '').strip()
    
    # Split into words, sort alphabetically, remove duplicates, and rejoin
    words = sorted(list(set(clean_str.split())))
    return " ".join(words).title()

# Apply the function to create a clean location column
df_combined['Cleaned_Location'] = df_combined['Location'].apply(standardize_location)

print("Locations successfully standardized!")
print("\n--- Top 10 Locations (Before Capping) ---")
print(df_combined['Cleaned_Location'].value_counts().head(10))

# From here, you can proceed to the Total Cost of Tenancy (TCT) extraction or directly to K-Means.