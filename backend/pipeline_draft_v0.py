import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load your capped dataset
df = pd.read_csv('lagos_rent_tct_extracted.csv')

# 2. Combine all text descriptions, attributes, and amenities into a single search space
df['Property_Profile'] = df['Raw_Description'].fillna('') + " " + \
                         df['Detailed_Attributes'].fillna('') + " " + \
                         df['Amenities'].fillna('')

# 3. Separate features (X) and our target variable (y)
# We use the 'TCT' column we just extracted as our target variable
X = df[['Cleaned_Location', 'Property_Profile']]
y = df['TCT']  # Make sure you're using the file where we calculated TCT!

# 4. Perform the strict 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train-Test Split Complete!")
print(f"Training Rows: {len(X_train)}")
print(f"Testing Rows: {len(X_test)}")