import pandas as pd
import joblib

def extract_model_decisions():
    print("--- EXTRACTING FEATURE IMPORTANCE FOR THESIS DEFENSE ---")
    
    # 1. Loading my fitted preprocessor and the trained baseline model
    preprocessor = joblib.load('feature_preprocessor.pkl')
    rf_model = joblib.load('rf_baseline_model_complete.pkl')
    
    # 2. Extracting all feature names dynamically from the preprocessor
    # This handles both the OneHotEncoded locations and the passthrough binary amenity columns
    raw_feature_names = preprocessor.get_feature_names_out()
    
    # 3. Cleaning up the names
    # sklearn attaches prefixes like 'location__' and 'remainder__' to the output names. 
    # I am splitting by '__' and taking the last part so my thesis tables look clean.
    clean_feature_names = [name.split('__')[-1] for name in raw_feature_names]
    
    # 4. Getting the mathematical importance weights assigned by the Random Forest
    importances = rf_model.feature_importances_
    
    # 5. Creating a readable DataFrame to map the names to their scores
    feature_df = pd.DataFrame({
        'Feature_Name': clean_feature_names,
        'Importance_Score': importances
    })
    
    # Sorting by the most important features first
    feature_df = feature_df.sort_values(by='Importance_Score', ascending=False).reset_index(drop=True)
    
    # Converting the raw score to a readable percentage format
    feature_df['Importance_Percentage'] = (feature_df['Importance_Score'] * 100).round(4).astype(str) + '%'
    
    # 6. Saving to CSV so I can embed this data directly into my documentation
    feature_df.to_csv('Model_Feature_Importances.csv', index=False)
    
    print("Success! Open 'Model_Feature_Importances.csv'")
    print("Top 10 Drivers of Rent in Lagos:")
    print(feature_df[['Feature_Name', 'Importance_Percentage']].head(10))

if __name__ == "__main__":
    extract_model_decisions()








# import pandas as pd
# import joblib

# def extract_model_decisions():
#     print("--- EXTRACTING FEATURE IMPORTANCE FOR THESIS DEFENSE ---")
    
#     # 1. Load the fitted preprocessor and the trained baseline model
#     preprocessor = joblib.load('feature_preprocessor.pkl')
#     rf_model = joblib.load('rf_baseline_model_complete.pkl')
    
#     # 2. Extract the exact names of the encoded locations
#     location_features = preprocessor.named_transformers_['location'].get_feature_names_out(['Cleaned_Location'])
    
#     # 3. Extract the exact top 400 words found by TF-IDF
#     text_features = preprocessor.named_transformers_['text'].get_feature_names_out()
    
#     # Combine them into one master list of all features
#     all_feature_names = list(location_features) + list(text_features)
    
#     # 4. Get the mathematical importance weights from the Random Forest
#     importances = rf_model.feature_importances_
    
#     # 5. Create a readable DataFrame
#     feature_df = pd.DataFrame({
#         'Feature_Name': all_feature_names,
#         'Importance_Score': importances
#     })
    
#     # Sort by the most important features first
#     feature_df = feature_df.sort_values(by='Importance_Score', ascending=False).reset_index(drop=True)
    
#     # Convert score to a readable percentage
#     feature_df['Importance_Percentage'] = (feature_df['Importance_Score'] * 100).round(4).astype(str) + '%'
    
#     # 6. Save to CSV so you can put it in your thesis tables
#     feature_df.to_csv('Model_Feature_Importances.csv', index=False)
    
#     print("Success! Open 'Model_Feature_Importances.csv'")
#     print("Top 5 Drivers of Rent in Lagos:")
#     print(feature_df[['Feature_Name', 'Importance_Percentage']].head(5))

# if __name__ == "__main__":
#     extract_model_decisions()