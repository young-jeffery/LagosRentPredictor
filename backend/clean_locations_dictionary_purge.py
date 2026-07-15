import pandas as pd

def clean_location_dictionary(input_path, output_path):
    # 1. Load the dictionary
    df = pd.read_csv(input_path)
    
    # 2. Define your column names (Change these if your CSV headers differ)
    main_col = 'Main_District'  
    sub_col = 'Sub_Location'    
    
    if main_col not in df.columns or sub_col not in df.columns:
        print(f"Error: Columns {main_col} or {sub_col} not found.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # 3. Map out the official hierarchy
    all_main_districts = set(df[main_col].dropna().unique())
    print(f"Detected {len(all_main_districts)} unique Main Districts.")
    
    # 4. The Purge Logic
    def is_valid_hierarchy(row):
        main = row[main_col]
        sub = row[sub_col]
        
        # If the sub-location is actually the name of a Main District...
        if sub in all_main_districts:
            # Keep it ONLY if it belongs to itself
            if sub == main:
                return True
            else:
                # This is a leakage bug (e.g., Sub: Ikorodu under Main: Ilashe)
                return False
                
        # Normal sub-locations are kept
        return True

    # 5. Apply the filter
    original_len = len(df)
    df_clean = df[df.apply(is_valid_hierarchy, axis=1)]
    purged_count = original_len - len(df_clean)
    
    # 6. Export the fixed dictionary
    df_clean.to_csv(output_path, index=False)
    print(f"Cleanup complete. Purged {purged_count} recursive locations.")
    print(f"Cleaned dictionary saved to: {output_path}")

if __name__ == "__main__":
    # Adjust filenames if necessary
    INPUT_DICT = 'Jiji_Official_Lagos_Dictionary.csv'
    OUTPUT_DICT = 'Jiji_Official_Lagos_Dictionary_Cleaned.csv'
    
    clean_location_dictionary(INPUT_DICT, OUTPUT_DICT)