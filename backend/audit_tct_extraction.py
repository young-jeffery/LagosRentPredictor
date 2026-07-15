import pandas as pd
import re

def audit_extraction(input_csv):
    print("--- INITIATING EXTRACTION AUDIT ---")
    df = pd.read_csv(input_csv)
    
    # Recreate the search space
    df['Combined_Text'] = df['Raw_Description'].fillna('') + " " + df['Detailed_Attributes'].fillna('')
    df['Combined_Text'] = df['Combined_Text'].str.lower()
    
    # Calculate the sum of all extracted fees
    df['Total_Fees_Extracted'] = df['Fee_Agency'] + df['Fee_Legal'] + df['Fee_Caution'] + df['Fee_Service']
    
    # Define keywords that strongly imply a fee breakdown exists
    # We include standard typos like 'agrement' or spacing issues like '10 %'
    keywords = ['agency', 'legal', 'agreement', 'agrement', 'caution', 'service charge', 'sc', '%']
    pattern = '|'.join(keywords)
    
    # The Trap: Find rows where text HAS keywords, but we extracted NOTHING (0.0)
    suspicious_mask = (df['Combined_Text'].str.contains(pattern, regex=True)) & (df['Total_Fees_Extracted'] == 0.0)
    df_suspicious = df[suspicious_mask]
    
    print(f"Total Rows Scanned: {len(df)}")
    print(f"Suspicious Rows (Potential False Negatives): {len(df_suspicious)}")
    
    # Export to a text file for easy reading
    output_file = 'TCT_Audit_Report.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== TCT EXTRACTION AUDIT REPORT ===\n")
        f.write(f"Found {len(df_suspicious)} properties that mention fees but extracted 0.0.\n")
        f.write("Review these to see if our Regex needs to account for weird agent typos.\n\n")
        
        for idx, row in df_suspicious.iterrows():
            f.write(f"[Row {idx}] Base Rent: {row['Base_Rent']}\n")
            # Print the first 300 characters of the text so you can spot the fee format
            f.write(f"Text Snippet: {row['Combined_Text'][:300]}...\n")
            f.write("-" * 50 + "\n")
            
    print(f"\nAudit complete. Open '{output_file}' to review the missed strings.")

if __name__ == "__main__":
    audit_extraction('lagos_rent_tct_extracted.csv')