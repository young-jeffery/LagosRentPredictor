import pandas as pd

def build_bulletproof_dataset(raw_data_path, raw_dict_path):
    print("--- STEP 1: INITIALIZING & PURGING NON-LAGOS DATA ---")
    
    # 1. Load data
    try:
        df_raw = pd.read_csv(raw_data_path)
        df_dict = pd.read_csv(raw_dict_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Check your location column name. Adjust if it's different in 'lagos_rent_cleaned_locations.csv'
    loc_col = 'Location' 
    if loc_col not in df_raw.columns:
        print(f"Error: Could not find '{loc_col}' column in raw dataset.")
        return
        
    original_len = len(df_raw)
    
    # Drop rows with missing locations
    df_raw = df_raw.dropna(subset=[loc_col])
    
    # Strictly enforce Lagos. Drop anything from other states (e.g., Jigawa, Edo).
    df_raw = df_raw[df_raw[loc_col].str.contains('Lagos', case=False, na=False)]
    
    print(f"Dropped {original_len - len(df_raw)} non-Lagos or empty rows.")
    print(f"Remaining pure Lagos rows: {len(df_raw)}")


    print("\n--- STEP 2: SANITIZING THE DICTIONARY ---")
    main_col = 'Main_District'
    sub_col = 'Sub_Location'
    
    # Drop empty sub-locations
    df_dict = df_dict.dropna(subset=[sub_col])
    
    # Clean strings for exact matching
    df_dict[sub_col] = df_dict[sub_col].str.strip().str.lower()
    df_dict[main_col] = df_dict[main_col].str.strip().str.title()
    
    # Identify unique Main Districts
    all_main_districts = set(df_dict[main_col].unique())
    
    # Function to catch the Ilashe/Lekki bug you noticed earlier
    def is_valid_hierarchy(row):
        sub = row[sub_col].title() # Check against Title Case Main Districts
        main = row[main_col]
        # If the sub-location is exactly the name of a Main District...
        if sub in all_main_districts:
            # Only keep it if it belongs to itself (e.g., Lekki under Lekki is fine)
            if sub != main:
                return False # Drop it! (e.g., Lekki under Ilashe)
        return True

    # Apply hierarchy fix
    df_dict = df_dict[df_dict.apply(is_valid_hierarchy, axis=1)]

    # Fix repeated sub-locations (1:1 mapping rule)
    dict_deduped = df_dict.drop_duplicates(subset=[sub_col], keep='first')
    
    print(f"Removed conflicting/recursive sub-locations. Dictionary ready.")
    
    # Export cleaned dictionary as TXT for easy reading
    with open('Cleaned_Lagos_Dictionary.txt', 'w', encoding='utf-8') as f:
        f.write("=== CLEANED LAGOS DICTIONARY ===\n\n")
        # Group by Main District to make it readable
        for main, group in dict_deduped.groupby(main_col):
            f.write(f"[{main}]\n")
            for sub in group[sub_col]:
                f.write(f"  - {sub.title()}\n")
            f.write("\n")
    print("Exported readable dictionary to 'Cleaned_Lagos_Dictionary.txt'")


    print("\n--- STEP 3: LONGEST-MATCH-FIRST MAPPING ---")
    # This logic naturally protects "Lekki Phase 1" from being mapped to "Phase 1"
    mapping_rules = {}
    for _, row in dict_deduped.iterrows():
        mapping_rules[row[sub_col]] = (row[sub_col].title(), row[main_col])
        
    sorted_sub_locations = sorted(mapping_rules.keys(), key=len, reverse=True)
    
    def map_location(raw_string):
        raw_string_lower = str(raw_string).lower()
        for sub_loc in sorted_sub_locations:
            if sub_loc in raw_string_lower:
                return mapping_rules[sub_loc]
        return ("Unknown", "Unknown")

    print("Mapping properties to official districts. This may take a moment...")
    mapped_results = df_raw[loc_col].apply(map_location)
    
    df_raw['Official_Sub_Location'] = [res[0] for res in mapped_results]
    df_raw['Main_District'] = [res[1] for res in mapped_results]
    
    unmapped_mask = df_raw['Main_District'] == 'Unknown'
    unmapped_count = unmapped_mask.sum()
    
    print(f"Successfully mapped: {len(df_raw) - unmapped_count} rows.")
    print(f"Failed to map (Unknown): {unmapped_count} rows.")
    
    # Save the final dataset as CSV for your Pandas models
    df_raw.to_csv('lagos_rent_strictly_mapped.csv', index=False)
    print("Exported mapped dataset to 'lagos_rent_strictly_mapped.csv'")
    
    # Export the unmapped raw strings to a TXT file so you can easily review what failed
    if unmapped_count > 0:
        unmapped_strings = df_raw.loc[unmapped_mask, loc_col].unique()
        with open('Unmapped_Locations.txt', 'w', encoding='utf-8') as f:
            f.write("=== STRINGS THAT FAILED TO MAP ===\n")
            f.write("Add missing variations of these to your raw dictionary.\n\n")
            for s in unmapped_strings:
                f.write(f"{s}\n")
        print("Exported failed strings to 'Unmapped_Locations.txt' for your review.")

if __name__ == "__main__":
    # Ensure these match your actual files
    RAW_DATA_FILE = 'lagos_rent_cleaned_locations.csv'
    RAW_DICT_FILE = 'Jiji_Official_Lagos_Dictionary.csv'
    
    build_bulletproof_dataset(RAW_DATA_FILE, RAW_DICT_FILE)