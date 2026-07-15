import pandas as pd

# Load the file generated from Phase A
df = pd.read_csv('lagos_rent_tct_extracted.csv')

# Filter for valid properties
valid_properties = df[df['TCT'] > 0]

print(f"Total properties before TCT filter: {len(df)}")
print(f"Total properties ready for training: {len(valid_properties)}")